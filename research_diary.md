# Research Diary

## Research log

| Experiment | Preprocessing | Architecture | Validation metric | Decision |
| --- | --- | --- | --- | --- |
| 1 | Lowercase word tokenization, fixed padding | Bidirectional RNN, 96-dimensional embeddings, 96 hidden units | Printed by the training script | Compare against LSTM |
| 2 | Same vocabulary and padding | Bidirectional LSTM, 96-dimensional embeddings, 96 hidden units | Printed by the training script | Select the best validation accuracy |
| 3 | Transfer learning with a pretrained BERT checkpoint | Documented notebook experiment | Run when `transformers` and model weights are available | Optional comparison |

The split is stratified and deterministic (`random_state=42`). The vocabulary is fitted only on the training split to prevent validation leakage. The fallback threshold is chosen from validation confidence and is never lower than 0.45. This makes low-confidence predictions return `Fallback Intent`, which is required for test phrases outside the training intents.

The serialized artifact stores the vocabulary, labels, sequence length, architecture, learned weights, and fallback threshold. A reviewer can load `model.pkl` with Python `pickle` and call `SavedModel.predict`.
