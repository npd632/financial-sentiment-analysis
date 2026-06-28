"""FinBERT inference helpers shared across training, evaluation, and demo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

LABEL2ID = {"neutral": 0, "positive": 1, "negative": 2}
ID2LABEL = {0: "neutral", 1: "positive", 2: "negative"}


def extract_cls_embeddings(
    texts: list[str],
    model_dir: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Extract FinBERT [CLS] embeddings (768-dim) for each text."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encodings = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encodings = {key: val.to(device) for key, val in encodings.items()}
            outputs = model.bert(**encodings)
            cls_batch = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_batch)

    return np.vstack(embeddings)


def predict_finbert_with_probs(
    texts: list[str],
    model_dir: str,
    max_length: int,
    batch_size: int,
) -> pd.DataFrame:
    """Run batched FinBERT inference; return labels and class probabilities."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    predicted_sentiment: list[str] = []
    prob_neutral: list[float] = []
    prob_positive: list[float] = []
    prob_negative: list[float] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encodings = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encodings = {key: val.to(device) for key, val in encodings.items()}
            logits = model(**encodings).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_ids = probs.argmax(axis=-1)

            for pred_id, row_probs in zip(pred_ids, probs):
                predicted_sentiment.append(ID2LABEL[int(pred_id)])
                prob_neutral.append(float(row_probs[0]))
                prob_positive.append(float(row_probs[1]))
                prob_negative.append(float(row_probs[2]))

    return pd.DataFrame(
        {
            "predicted_sentiment": predicted_sentiment,
            "prob_neutral": prob_neutral,
            "prob_positive": prob_positive,
            "prob_negative": prob_negative,
        }
    )


def predict_finbert(
    texts: list[str],
    model_dir: str,
    max_length: int,
    batch_size: int,
) -> list[str]:
    """Run batched FinBERT inference and return sentiment label strings."""
    results = predict_finbert_with_probs(texts, model_dir, max_length, batch_size)
    return results["predicted_sentiment"].tolist()
