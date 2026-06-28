#!/usr/bin/env python3
"""FinBERT fine-tuning for 3-class sentiment (README section 5.2)."""

import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_config
from finbert_inference import ID2LABEL, LABEL2ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TEXT_COL = "cleaned_text"
LABEL_COL = "Sentiment"


class SentimentDataset(torch.utils.data.Dataset):
    """PyTorch dataset for Hugging Face Trainer."""

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """Metrics callback for Trainer (best checkpoint selected by macro_f1)."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    _, _, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
    }


def train_finbert(config: dict) -> None:
    """Fine-tune FinBERT on preprocessed PhraseBank and save model + tokenizer."""
    data_cfg = config["data"]
    finbert_cfg = config["models"]["finbert"]

    input_path = data_cfg["phrasebank_clean_path"]
    train_split = data_cfg["train_split"]
    random_seed = data_cfg["random_seed"]
    save_dir = finbert_cfg["save_path"]

    pretrained_model = finbert_cfg["pretrained_model"]
    max_length = finbert_cfg["max_length"]
    batch_size = finbert_cfg["batch_size"]
    epochs = finbert_cfg["epochs"]
    learning_rate = float(finbert_cfg["learning_rate"])

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Preprocessed PhraseBank not found: {input_path}. "
            "Run `python src/preprocess.py` first."
        )

    logger.info("Loading %s", input_path)
    df = pd.read_csv(input_path)

    for col in (TEXT_COL, LABEL_COL):
        if col not in df.columns:
            raise ValueError(
                f"Required column {col!r} not found. Available: {list(df.columns)}"
            )

    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[df[TEXT_COL].astype(str).str.strip() != ""]
    df["label_id"] = df[LABEL_COL].map(LABEL2ID)

    if df["label_id"].isna().any():
        unknown = df.loc[df["label_id"].isna(), LABEL_COL].unique()
        raise ValueError(f"Unknown sentiment labels: {unknown}")

    logger.info("Training on %d samples", len(df))

    logger.info(
        "Splitting dataset: train=%.0f%%, test=%.0f%%, random_state=%d",
        train_split * 100,
        (1 - train_split) * 100,
        random_seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        df[TEXT_COL],
        df["label_id"],
        test_size=(1.0 - train_split),
        random_state=random_seed,
        stratify=df["label_id"],
    )

    logger.info("Loading tokenizer: %s", pretrained_model)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    logger.info("Tokenizing datasets (max_length=%d)", max_length)
    train_encodings = tokenizer(
        list(x_train), truncation=True, padding=True, max_length=max_length
    )
    test_encodings = tokenizer(
        list(x_test), truncation=True, padding=True, max_length=max_length
    )

    train_dataset = SentimentDataset(train_encodings, list(y_train))
    test_dataset = SentimentDataset(test_encodings, list(y_test))

    logger.info("Loading pretrained model: %s", pretrained_model)
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    os.makedirs(save_dir, exist_ok=True)

    use_fp16 = torch.cuda.is_available()
    logger.info("Mixed precision (fp16): %s", use_fp16)

    training_args = TrainingArguments(
        output_dir=os.path.join(save_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=random_seed,
        fp16=use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting fine-tuning (%d epochs, batch_size=%d)...", epochs, batch_size)
    trainer.train()

    logger.info("Evaluating on held-out test set...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    accuracy = accuracy_score(labels, preds)
    _, _, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    logger.info(
        "FinBERT — accuracy: %.4f, macro F1: %.4f, weighted F1: %.4f",
        accuracy,
        f1_macro,
        f1_weighted,
    )

    logger.info("Saving model and tokenizer to %s", save_dir)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)


def main(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    train_finbert(config)
    logger.info("FinBERT training completed.")


if __name__ == "__main__":
    main()
