"""Train and use RNN, LSTM, and optional BERT intent classifiers.

Run ``python intent_classifier.py`` from the project root after placing the
assignment files in ``data/``. The script writes ``model.pkl`` and
``intents.csv``.
"""

from __future__ import annotations

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

if __name__ == "__main__":
    sys.modules.setdefault("intent_classifier", sys.modules[__name__])

SEED = 42
FALLBACK_INTENT = "Fallback Intent"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_column(frame: pd.DataFrame, candidates: Iterable[str], kind: str) -> str:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    if kind == "text":
        for column in frame.columns:
            if frame[column].dtype == "object":
                return column
    raise ValueError(f"Could not identify the {kind} column. Found: {list(frame.columns)}")


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    train_path = next((data_dir / name for name in ("intents_train.csv", "train.csv") if (data_dir / name).exists()), None)
    test_path = next((data_dir / name for name in ("intents_test.csv", "test.csv") if (data_dir / name).exists()), None)
    if train_path is None or test_path is None:
        raise FileNotFoundError("Put intents_train.csv and intents_test.csv in the data/ directory.")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    text_column = find_column(train, ("text", "phrase", "utterance", "query", "sentence", "question"), "text")
    label_column = find_column(train, ("intent", "label", "category", "class"), "label")
    test_text_column = text_column if text_column in test else find_column(test, ("text", "phrase", "utterance", "query", "sentence", "question"), "text")
    train = train[[text_column, label_column]].dropna().rename(columns={text_column: "text", label_column: "label"})
    test = test[[test_text_column]].fillna("").rename(columns={test_text_column: "text"})
    train["text"] = train["text"].astype(str)
    test["text"] = test["text"].astype(str)
    train["label"] = train["label"].astype(str)
    return train, test, text_column, label_column


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


if __name__ == "__main__":
    Vocabulary.__module__ = "intent_classifier"


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


@dataclass
class SavedModel:
    architecture: str
    vocabulary: Vocabulary
    labels: list[str]
    max_length: int
    model_state: dict
    fallback_threshold: float
    embedding_dim: int = 96
    hidden_dim: int = 96

    def predict(self, texts: Iterable[str]) -> tuple[list[str], np.ndarray]:
        model = SequenceClassifier(len(self.vocabulary.token_to_id), len(self.labels), self.architecture, self.embedding_dim, self.hidden_dim)
        model.load_state_dict(self.model_state)
        model.eval()
        encoded = torch.tensor([self.vocabulary.encode(text, self.max_length) for text in texts], dtype=torch.long)
        with torch.no_grad():
            probabilities = torch.softmax(model(encoded), dim=1).numpy()
        indexes = probabilities.argmax(axis=1)
        predictions = [self.labels[index] if probabilities[row, index] >= self.fallback_threshold else FALLBACK_INTENT for row, index in enumerate(indexes)]
        return predictions, probabilities


if __name__ == "__main__":
    SavedModel.__module__ = "intent_classifier"


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
    train_frame, test_frame, _, _ = load_data(data_dir)
    labels, label_ids = encode_labels(train_frame["label"])
    stratify_labels = train_frame["label"] if train_frame["label"].value_counts().min() >= 2 else None
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        train_frame["text"].tolist(), [label_ids[label] for label in train_frame["label"]],
        test_size=0.2, random_state=SEED, stratify=stratify_labels,
    )
    vocabulary = Vocabulary()
    vocabulary.fit(train_texts)
    max_length = min(40, max(8, max(len(tokenize(text)) for text in train_frame["text"])))
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
    artifact = SavedModel(architecture, vocabulary, labels, max_length, selected_model.state_dict(), fallback_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(artifact, handle)
    predictions, _ = artifact.predict(test_frame["text"].tolist())
    pd.DataFrame({"intent": predictions}).to_csv(output_dir / "intents.csv", index=False)
    validation_predictions = validation_probabilities.argmax(axis=1)
    print(f"Selected model: {architecture.upper()}")
    print(f"Validation accuracy: {validation_accuracy:.3f}")
    print(classification_report(validation_labels, validation_predictions, labels=list(range(len(labels))), target_names=labels, zero_division=0))
    print(f"Wrote {output_dir / 'model.pkl'} and {output_dir / 'intents.csv'}")


def predict(model_path: Path, input_path: Path, output_path: Path) -> None:
    with model_path.open("rb") as handle:
        artifact: SavedModel = pickle.load(handle)
    frame = pd.read_csv(input_path)
    text_column = find_column(frame, ("text", "phrase", "utterance", "query", "sentence", "question"), "text")
    predictions, _ = artifact.predict(frame[text_column].fillna("").astype(str).tolist())
    pd.DataFrame({"intent": predictions}).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--predict", action="store_true", help="Use an existing model.pkl instead of training")
    args = parser.parse_args()
    if args.predict:
        predict(args.output_dir / "model.pkl", args.data_dir / "intents_test.csv", args.output_dir / "intents.csv")
    else:
        train(args.data_dir, args.output_dir, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()