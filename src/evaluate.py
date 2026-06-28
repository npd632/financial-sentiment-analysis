#!/usr/bin/env python3
"""Model evaluation: sentiment (Phase 3) and market impact (Phase 4)."""

import argparse
import json
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TEXT_COL = "cleaned_text"
LABEL_COL = "Sentiment"
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {"neutral": 0, "positive": 1, "negative": 2}
ID2LABEL = {0: "neutral", 1: "positive", 2: "negative"}
SENTIMENT_TO_DIRECTION = {"positive": "Up", "negative": "Down"}
DIRECTION_LABELS = ["Up", "Down"]


def load_phrasebank_test_split(config: dict) -> tuple[pd.Series, pd.Series]:
    """Load PhraseBank and return the same stratified test split used in training."""
    data_cfg = config["data"]
    input_path = data_cfg["phrasebank_clean_path"]

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Preprocessed PhraseBank not found: {input_path}. "
            "Run `python src/preprocess.py` first."
        )

    df = pd.read_csv(input_path)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[df[TEXT_COL].astype(str).str.strip() != ""]

    _, x_test, _, y_test = train_test_split(
        df[TEXT_COL],
        df[LABEL_COL],
        test_size=(1.0 - data_cfg["train_split"]),
        random_state=data_cfg["random_seed"],
        stratify=df[LABEL_COL],
    )
    logger.info("Held-out test set: %d samples", len(x_test))
    return x_test, y_test


def build_metrics_dict(
    y_true: list | np.ndarray | pd.Series,
    y_pred: list | np.ndarray,
    labels: list[str] | None = None,
) -> dict:
    """Compute accuracy, macro/weighted averages, and per-class metrics."""
    if labels is None:
        labels = sorted(set(y_true))

    accuracy = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    prec_class, rec_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class = {}
    for idx, label in enumerate(labels):
        per_class[label] = {
            "precision": float(prec_class[idx]),
            "recall": float(rec_class[idx]),
            "f1_score": float(f1_class[idx]),
            "support": int(support_class[idx]),
        }

    return {
        "accuracy": float(accuracy),
        "macro_avg": {
            "precision": float(prec_macro),
            "recall": float(rec_macro),
            "f1_score": float(f1_macro),
        },
        "weighted_avg": {
            "precision": float(prec_weighted),
            "recall": float(rec_weighted),
            "f1_score": float(f1_weighted),
        },
        "per_class": per_class,
    }


def plot_confusion_matrix(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    labels: list[str],
    output_path: str,
    title: str,
) -> None:
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", output_path)


def evaluate_baselines(
    config: dict,
    x_test: pd.Series,
    y_test: pd.Series,
    figures_path: str,
) -> dict:
    """Evaluate saved TF-IDF + Naive Bayes / SVM models."""
    baseline_cfg = config["models"]["baseline"]
    save_dir = baseline_cfg["save_path"]

    vectorizer_path = os.path.join(save_dir, "tfidf_vectorizer.pkl")
    model_paths = {
        "naive_bayes": os.path.join(save_dir, "naive_bayes.pkl"),
        "svm": os.path.join(save_dir, "svm.pkl"),
    }

    for path in [vectorizer_path, *model_paths.values()]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Baseline artifact not found: {path}. "
                "Run `python src/train_baseline.py` first."
            )

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    x_test_tfidf = vectorizer.transform(x_test)
    metrics = {}

    for name, model_path in model_paths.items():
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        y_pred = model.predict(x_test_tfidf)
        metrics[name] = build_metrics_dict(y_test, y_pred, labels=SENTIMENT_LABELS)

        macro_f1 = metrics[name]["macro_avg"]["f1_score"]
        logger.info(
            "%s — accuracy: %.4f, macro F1: %.4f",
            name,
            metrics[name]["accuracy"],
            macro_f1,
        )

        plot_confusion_matrix(
            y_test,
            y_pred,
            SENTIMENT_LABELS,
            os.path.join(figures_path, f"confusion_matrix_{name}.png"),
            title=f"{name.replace('_', ' ').title()} — Confusion Matrix",
        )

    return metrics


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


def evaluate_finbert(
    config: dict,
    x_test: pd.Series,
    y_test: pd.Series,
    figures_path: str,
) -> dict:
    """Evaluate the fine-tuned FinBERT model."""
    finbert_cfg = config["models"]["finbert"]
    model_dir = finbert_cfg["save_path"]

    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(
            f"FinBERT model not found in {model_dir}. "
            "Run `python src/train_finbert.py` first."
        )

    y_pred = predict_finbert(
        list(x_test),
        model_dir=model_dir,
        max_length=finbert_cfg["max_length"],
        batch_size=finbert_cfg["batch_size"],
    )

    metrics = {"finbert": build_metrics_dict(y_test, y_pred, labels=SENTIMENT_LABELS)}
    macro_f1 = metrics["finbert"]["macro_avg"]["f1_score"]
    logger.info(
        "finbert — accuracy: %.4f, macro F1: %.4f",
        metrics["finbert"]["accuracy"],
        macro_f1,
    )

    plot_confusion_matrix(
        y_test,
        y_pred,
        SENTIMENT_LABELS,
        os.path.join(figures_path, "confusion_matrix_finbert.png"),
        title="FinBERT — Confusion Matrix",
    )

    return metrics


def select_best_model(all_metrics: dict) -> str:
    """Select best model by macro F1 (README primary selection metric)."""
    candidates = {
        name: scores["macro_avg"]["f1_score"]
        for name, scores in all_metrics.items()
        if "macro_avg" in scores
    }
    return max(candidates, key=candidates.get)


def write_json(path: str, data: dict) -> None:
    """Write metrics dictionary to JSON."""
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info("Saved metrics to %s", path)


def evaluate_sentiment(config: dict) -> None:
    """Phase 3: evaluate all sentiment models on the PhraseBank test set."""
    eval_cfg = config["evaluation"]
    figures_path = eval_cfg["figures_path"]

    x_test, y_test = load_phrasebank_test_split(config)

    baseline_metrics = evaluate_baselines(config, x_test, y_test, figures_path)
    finbert_metrics = evaluate_finbert(config, x_test, y_test, figures_path)

    write_json(eval_cfg["metrics_module1_path"], baseline_metrics)
    write_json(eval_cfg["metrics_module2_path"], finbert_metrics)

    merged = {**baseline_metrics, **finbert_metrics}
    merged["best_model"] = select_best_model(merged)
    merged["selection_metric"] = "macro_f1"
    write_json(eval_cfg["metrics_path"], merged)

    logger.info("Best model (macro F1): %s", merged["best_model"])


def compute_directional_metrics(
    df: pd.DataFrame,
) -> dict:
    """Compute agreement rate, confusion matrix, and MCC for Up/Down rows."""
    directional = df[df["predicted_sentiment"] != "neutral"].copy()
    directional["predicted_direction"] = directional["predicted_sentiment"].map(
        SENTIMENT_TO_DIRECTION
    )

    y_true = (directional["price_direction"] == "Up").astype(int)
    y_pred = (directional["predicted_direction"] == "Up").astype(int)

    agreement = float((directional["predicted_direction"] == directional["price_direction"]).mean())
    cm = confusion_matrix(
        directional["price_direction"],
        directional["predicted_direction"],
        labels=DIRECTION_LABELS,
    )

    return {
        "directional_sample_size": int(len(directional)),
        "directional_agreement_rate": agreement,
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": {
            "labels": DIRECTION_LABELS,
            "matrix": cm.tolist(),
        },
    }


def plot_direction_heatmap(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """Save heatmap of predicted vs actual price direction (non-neutral only)."""
    directional = df[df["predicted_sentiment"] != "neutral"].copy()
    directional["predicted_direction"] = directional["predicted_sentiment"].map(
        SENTIMENT_TO_DIRECTION
    )

    cm = confusion_matrix(
        directional["price_direction"],
        directional["predicted_direction"],
        labels=DIRECTION_LABELS,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=DIRECTION_LABELS,
        yticklabels=DIRECTION_LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted direction (from sentiment)")
    ax.set_ylabel("Actual price direction")
    ax.set_title("Sentiment vs Price Direction (non-neutral predictions)")
    fig.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved direction heatmap to %s", output_path)


def evaluate_market(config: dict) -> None:
    """Phase 4: measure alignment between FinBERT sentiment and same-day price direction."""
    data_cfg = config["data"]
    finbert_cfg = config["models"]["finbert"]
    eval_cfg = config["evaluation"]

    news_path = data_cfg["news_subset_path"]
    aligned_path = data_cfg["aligned_news_prices_path"]
    model_dir = finbert_cfg["save_path"]

    for path in (news_path, aligned_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: {path}")

    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(
            f"FinBERT model not found in {model_dir}. "
            "Run `python src/train_finbert.py` first."
        )

    logger.info("Loading news subset from %s", news_path)
    news = pd.read_csv(news_path)
    logger.info("Running FinBERT inference on %d headlines...", len(news))

    predictions = predict_finbert_with_probs(
        list(news["cleaned_text"]),
        model_dir=model_dir,
        max_length=finbert_cfg["max_length"],
        batch_size=finbert_cfg["batch_size"],
    )
    news = pd.concat([news.reset_index(drop=True), predictions], axis=1)

    logger.info("Loading aligned news/prices from %s", aligned_path)
    aligned = pd.read_csv(aligned_path)

    merged = aligned.merge(
        news[
            [
                "headline",
                "stock",
                "date",
                "predicted_sentiment",
                "prob_negative",
                "prob_neutral",
                "prob_positive",
            ]
        ],
        left_on=["headline", "stock", "news_datetime"],
        right_on=["headline", "stock", "date"],
        how="inner",
    )
    logger.info("Merged %d rows with price data and sentiment predictions", len(merged))

    neutral_count = int((merged["predicted_sentiment"] == "neutral").sum())
    overall = compute_directional_metrics(merged)

    merged["year"] = pd.to_datetime(merged["trading_date"]).dt.year
    breakdown_by_year = {}
    for year in (2018, 2019, 2020):
        year_df = merged[merged["year"] == year]
        if year_df.empty:
            breakdown_by_year[str(year)] = {
                "directional_sample_size": 0,
                "directional_agreement_rate": None,
                "matthews_correlation_coefficient": None,
            }
            continue
        year_metrics = compute_directional_metrics(year_df)
        breakdown_by_year[str(year)] = {
            "directional_sample_size": year_metrics["directional_sample_size"],
            "directional_agreement_rate": year_metrics["directional_agreement_rate"],
            "matthews_correlation_coefficient": year_metrics[
                "matthews_correlation_coefficient"
            ],
        }

    metrics = {
        "total_news_rows": int(len(news)),
        "aligned_rows": int(len(aligned)),
        "merged_rows": int(len(merged)),
        "neutral_predictions_excluded": neutral_count,
        **overall,
        "breakdown_by_year": breakdown_by_year,
        "limitation": (
            "Measures same-day open-to-close co-movement between headline sentiment "
            "and price direction; does not establish causal impact of news on prices."
        ),
    }

    write_json(eval_cfg["metrics_module3_path"], metrics)
    plot_direction_heatmap(
        merged,
        os.path.join(eval_cfg["figures_path"], "sentiment_vs_price.png"),
    )

    logger.info(
        "Directional agreement: %.4f (n=%d, neutral excluded=%d, MCC=%.4f)",
        overall["directional_agreement_rate"],
        overall["directional_sample_size"],
        neutral_count,
        overall["matthews_correlation_coefficient"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument(
        "--module",
        choices=["sentiment", "market"],
        default="sentiment",
        help="Evaluation module to run (default: sentiment)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.module == "sentiment":
        evaluate_sentiment(config)
    else:
        evaluate_market(config)

    logger.info("Evaluation completed.")


if __name__ == "__main__":
    main()
