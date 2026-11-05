import argparse
import pickle
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
FALLBACK_INTENT = "Fallback Intent"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Vocabulary:
    def __init__(self, min_frequency: int = 1, max_size: int = 12000):
        self.min_frequency = min_frequency
        self.max_size = max_size
        self.token_to_id = {"<pad>": 0, "<unk>": 1}

    def fit(self, texts: Iterable[str]) -> None:
        counts: dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        ranked = sorted(((count, token) for token, count in counts.items() if count >= self.min_frequency), reverse=True)
        for _, token in ranked[: self.max_size - 2]:
            self.token_to_id[token] = len(self.token_to_id)

    def encode(self, text: str, max_length: int) -> list[int]:
        ids = [self.token_to_id.get(token, 1) for token in tokenize(text)][:max_length]
        return ids + [0] * (max_length - len(ids))



class SequenceClassifier(nn.Module):
    def __init__(self, vocabulary_size: int, class_count: int, architecture: str, embedding_dim: int = 96, hidden_dim: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        recurrent = nn.RNN if architecture == "rnn" else nn.LSTM
        self.recurrent = recurrent(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.25)
        self.output = nn.Linear(hidden_dim * 2, class_count)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, state = self.recurrent(self.embedding(tokens))
        hidden = state[0] if isinstance(state, tuple) else state
        pooled = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.output(self.dropout(pooled))



def encode_labels(labels: pd.Series) -> tuple[list[str], dict[str, int]]:
    names = sorted(labels.unique().tolist())
    return names, {name: index for index, name in enumerate(names)}


def train_architecture(train_texts: list[str], train_labels: list[int], validation_texts: list[str], validation_labels: list[int], vocabulary: Vocabulary, architecture: str, max_length: int, epochs: int, batch_size: int, class_count: int) -> tuple[SequenceClassifier, float]:
    model = SequenceClassifier(len(vocabulary.token_to_id), class_count, architecture)
    train_x = torch.tensor([vocabulary.encode(text, max_length) for text in train_texts], dtype=torch.long)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    validation_x = torch.tensor([vocabulary.encode(text, max_length) for text in validation_texts], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    best_accuracy = -1.0
    best_state = None
    for _ in range(epochs):
        model.train()
        for tokens, targets in loader:
            optimizer.zero_grad()
            loss = loss_function(model(tokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_predictions = model(validation_x).argmax(dim=1).numpy()
        current_accuracy = accuracy_score(validation_labels, validation_predictions)
        if current_accuracy >= best_accuracy:
            best_accuracy = current_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_accuracy


def train(data_dir: Path, output_dir: Path, epochs: int, batch_size: int) -> None:
    seed_everything()
    train = pd.read_csv("data/intents_train.csv")
    test = pd.read_csv("data/intents_test.csv")
    labels, label_ids = encode_labels(train["intent"])
    stratify_labels = train["intent"] if train["intent"].value_counts().min() >= 2 else None
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        train["text"].tolist(), [label_ids[label] for label in train["intent"]],
        test_size=0.2, random_state=SEED, stratify=stratify_labels,
    )
    vocabulary = Vocabulary()
    vocabulary.fit(train_texts)
    max_length = min(40, max(8, max(len(tokenize(text)) for text in train["text"])))
    results = {}
    for architecture in ("rnn", "lstm"):
        model, validation_accuracy = train_architecture(train_texts, train_labels, validation_texts, validation_labels, vocabulary, architecture, max_length, epochs, batch_size, len(labels))
        results[architecture] = (model, validation_accuracy)
        print(f"{architecture.upper()} validation accuracy: {validation_accuracy:.3f}")
    architecture = max(results, key=lambda name: results[name][1])
    selected_model, validation_accuracy = results[architecture]
    selected_model.eval()
    validation_x = torch.tensor([vocabulary.encode(text, max_length) for text in validation_texts], dtype=torch.long)
    with torch.no_grad():
        validation_probabilities = torch.softmax(selected_model(validation_x), dim=1).numpy()
    correct_confidences = validation_probabilities[np.arange(len(validation_labels)), validation_labels]
    fallback_threshold = float(max(0.45, np.quantile(correct_confidences, 0.05)))
    indexes = validation_probabilities.argmax(axis=1)
    predictions = [labels[index] if validation_probabilities[row, index] >= fallback_threshold else FALLBACK_INTENT for row, index in enumerate(indexes)]
    pd.DataFrame({"intent": predictions}).to_csv(output_dir / "intents.csv", index=False)
    print(f"Selected model: {architecture.upper()}")
    print(f"Validation accuracy: {validation_accuracy:.3f}")
    print(f"Wrote {output_dir / 'intents.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()


    