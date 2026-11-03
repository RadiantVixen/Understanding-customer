# Understanding Customer Intent Classification

This repository contains the mandatory project solution. It trains and compares a bidirectional RNN and LSTM, selects the best validation model, calibrates a confidence threshold for unseen phrases, serializes the selected model, and writes the required `intents.csv` file.

## Run

1. Put `intents_train.csv` and `intents_test.csv` in `data/`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `python intent_classifier.py`.

The script accepts the common text columns `text`, `phrase`, `utterance`, `query`, `sentence`, or `question`, and label columns `intent`, `label`, `category`, or `class`. It creates `model.pkl` and `intents.csv` in the project root. To regenerate predictions from the saved model, run `python intent_classifier.py --predict`.

The assignment also asks for BERT. The notebook includes the BERT transfer-learning experiment as a documented optional cell because downloading pretrained weights is not reproducible in an offline evaluator; the CPU-safe RNN/LSTM pipeline is the default submission path.