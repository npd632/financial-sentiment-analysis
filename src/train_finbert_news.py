#!/usr/bin/env python3
"""Fine-tune FinBERT on news headlines: direction + sentiment distillation (v2 Stage 1B)."""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_price_dataset import build_labeled_frame
from finbert_inference import ID2LABEL, LABEL2ID, predict_finbert_with_probs
from preprocess import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class BertMultiTaskForNews(BertPreTrainedModel):
    """Shared BERT with sentiment (3-class) and direction (2-class) heads."""

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.sentiment_classifier = torch.nn.Linear(config.hidden_size, 3)
        self.direction_classifier = torch.nn.Linear(config.hidden_size, 2)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        direction_labels=None,
        teacher_probs=None,
        direction_weight=1.0,
        sentiment_weight=0.5,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        sentiment_logits = self.sentiment_classifier(pooled)
        direction_logits = self.direction_classifier(pooled)

        loss = None
        if direction_labels is not None:
            dir_loss = F.cross_entropy(direction_logits, direction_labels)
            if teacher_probs is not None:
                log_probs = F.log_softmax(sentiment_logits, dim=-1)
                sent_loss = F.kl_div(
                    log_probs,
                    teacher_probs,
                    reduction="batchmean",
                )
                loss = direction_weight * dir_loss + sentiment_weight * sent_loss
            else:
                loss = dir_loss

        return {
            "loss": loss,
            "sentiment_logits": sentiment_logits,
            "direction_logits": direction_logits,
        }


class NewsMultiTaskDataset(Dataset):
    def __init__(
        self,
        encodings: dict,
        direction_labels: list[int],
        teacher_probs: np.ndarray,
    ):
        self.encodings = encodings
        self.direction_labels = direction_labels
        self.teacher_probs = teacher_probs

    def __len__(self) -> int:
        return len(self.direction_labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["direction_labels"] = torch.tensor(self.direction_labels[idx], dtype=torch.long)
        item["teacher_probs"] = torch.tensor(self.teacher_probs[idx], dtype=torch.float)
        return item


class MultiTaskTrainer(Trainer):
    def __init__(self, *args, direction_weight=1.0, sentiment_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction_weight = direction_weight
        self.sentiment_weight = sentiment_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        direction_labels = inputs.pop("direction_labels")
        teacher_probs = inputs.pop("teacher_probs")
        outputs = model(
            **inputs,
            direction_labels=direction_labels,
            teacher_probs=teacher_probs,
            direction_weight=self.direction_weight,
            sentiment_weight=self.sentiment_weight,
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


def export_sentiment_checkpoint(
    multitask_model: BertMultiTaskForNews,
    save_dir: str,
    tokenizer: BertTokenizer,
    init_checkpoint: str,
) -> None:
    """Save BertForSequenceClassification compatible with finbert_inference."""
    base = BertForSequenceClassification.from_pretrained(
        init_checkpoint,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    base.bert.load_state_dict(multitask_model.bert.state_dict())
    base.classifier.load_state_dict(multitask_model.sentiment_classifier.state_dict())
    os.makedirs(save_dir, exist_ok=True)
    base.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info("Exported sentiment checkpoint to %s", save_dir)


def train_finbert_news(config: dict) -> None:
    data_cfg = config["data"]
    finbert_cfg = config["models"]["finbert"]
    news_cfg = config["models"]["finbert_news"]

    init_checkpoint = news_cfg["init_checkpoint"]
    save_dir = news_cfg["save_path"]
    max_length = news_cfg.get("max_length", finbert_cfg["max_length"])
    batch_size = news_cfg["batch_size"]
    epochs = news_cfg["epochs"]
    learning_rate = float(news_cfg["learning_rate"])
    direction_weight = float(news_cfg["direction_loss_weight"])
    sentiment_weight = float(news_cfg["sentiment_distill_weight"])

    if not os.path.exists(os.path.join(init_checkpoint, "config.json")):
        raise FileNotFoundError(
            f"Init checkpoint not found: {init_checkpoint}. Run train_finbert.py first."
        )

    logger.info("Building labeled news frame for multi-task training...")
    df = build_labeled_frame(config)
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    train_end = pd.Timestamp(data_cfg["train_end_date"])
    df = df[df["trading_date"] <= train_end].copy()
    logger.info("Training rows (<= %s): %d", train_end.date(), len(df))

    teacher_dir = finbert_cfg["save_path"]
    logger.info("Generating teacher sentiment probabilities from %s", teacher_dir)
    teacher = predict_finbert_with_probs(
        list(df["cleaned_text"]),
        model_dir=teacher_dir,
        max_length=max_length,
        batch_size=batch_size,
    )
    teacher_probs = teacher[["prob_neutral", "prob_positive", "prob_negative"]].values.astype(
        np.float32
    )
    direction_labels = (df["forward_direction"] == "Up").astype(int).tolist()

    tokenizer = BertTokenizer.from_pretrained(init_checkpoint)
    encodings = tokenizer(
        list(df["cleaned_text"]),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    dataset = NewsMultiTaskDataset(encodings, direction_labels, teacher_probs)

    base_model = BertForSequenceClassification.from_pretrained(init_checkpoint)
    multitask = BertMultiTaskForNews(base_model.config)
    multitask.bert.load_state_dict(base_model.bert.state_dict())
    multitask.sentiment_classifier.load_state_dict(base_model.classifier.state_dict())

    os.makedirs(save_dir, exist_ok=True)
    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=os.path.join(save_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=data_cfg["random_seed"],
        fp16=use_fp16,
    )

    trainer = MultiTaskTrainer(
        model=multitask,
        args=training_args,
        train_dataset=dataset,
        direction_weight=direction_weight,
        sentiment_weight=sentiment_weight,
    )

    logger.info(
        "Starting news multi-task fine-tune (%d epochs, %d rows)...",
        epochs,
        len(dataset),
    )
    trainer.train()

    export_sentiment_checkpoint(multitask, save_dir, tokenizer, init_checkpoint)
    logger.info("News-adapted FinBERT saved to %s", save_dir)


def main(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    train_finbert_news(config)


if __name__ == "__main__":
    main()
