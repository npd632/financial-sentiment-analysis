#!/usr/bin/env python3
"""Model evaluation: sentiment (Phase 3) and market impact (Phase 4)."""

import argparse
import json
import logging
import os
import joblib
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finbert_inference import predict_finbert, predict_finbert_with_probs
from train_price_model import prepare_matrices, temporal_split
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
DIRECTION_LABELS = ["Down", "Up"]


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


from finbert_inference import predict_finbert, predict_finbert_with_probs
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


def transform_with_pipeline(
    pipeline: dict,
    x_tab: np.ndarray,
    x_cls: np.ndarray,
) -> np.ndarray:
    """Apply fitted scalers and PCA to feature blocks."""
    x_tab_s = pipeline["scaler_tabular"].transform(x_tab)
    x_cls_p = pipeline["pca"].transform(x_cls)
    x_cls_s = pipeline["scaler_cls"].transform(x_cls_p)
    return np.hstack([x_tab_s, x_cls_s])


def direction_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob_up: np.ndarray) -> dict:
    """Compute classification and calibration metrics for Up=1 encoding."""
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    correct = y_pred == y_true
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob_up)),
        "mean_confidence": float(np.maximum(y_prob_up, 1 - y_prob_up).mean()),
        "mean_confidence_correct": float(np.maximum(y_prob_up, 1 - y_prob_up)[correct].mean())
        if correct.any()
        else None,
        "mean_confidence_incorrect": float(np.maximum(y_prob_up, 1 - y_prob_up)[~correct].mean())
        if (~correct).any()
        else None,
        "per_class": {
            "Down": {
                "precision": float(prec[0]),
                "recall": float(rec[0]),
                "f1_score": float(f1[0]),
                "support": int(support[0]),
            },
            "Up": {
                "precision": float(prec[1]),
                "recall": float(rec[1]),
                "f1_score": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": {
            "labels": DIRECTION_LABELS,
            "matrix": cm.tolist(),
        },
    }


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob_up: np.ndarray,
    output_path: str,
) -> None:
    """Save reliability diagram for P(Up)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob_up, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted P(Up)")
    ax.set_ylabel("Fraction of Up outcomes")
    ax.set_title("Price Direction Calibration Curve (test set)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved calibration curve to %s", output_path)


def evaluate_market(config: dict) -> None:
    """Phase 4: evaluate Stage 2 calibrated price-direction model on temporal test set."""
    data_cfg = config["data"]
    price_cfg = config["models"]["price_direction"]
    eval_cfg = config["evaluation"]

    dataset_path = data_cfg["price_model_dataset_path"]
    pipeline_path = os.path.join(price_cfg["save_path"], "pipeline.pkl")

    for path in (dataset_path, pipeline_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required artifact not found: {path}. "
                "Run build_price_dataset.py and train_price_model.py first."
            )

    df = pd.read_parquet(dataset_path)
    train_df, test_df = temporal_split(
        df,
        train_end=data_cfg["price_train_end_date"],
        test_start=data_cfg["price_test_start_date"],
    )
    logger.info("Evaluating on temporal test set: %d rows (2020+)", len(test_df))

    pipeline = joblib.load(pipeline_path)
    x_tab_test, x_cls_test, y_test = prepare_matrices(test_df)
    x_test = transform_with_pipeline(pipeline, x_tab_test, x_cls_test)

    calibrator = pipeline["calibrator"]
    y_pred = calibrator.predict(x_test)
    y_prob_up = calibrator.predict_proba(x_test)[:, 1]

    stage2 = direction_metrics(y_test, y_pred, y_prob_up)

    # Baselines on test set
    y_pred_always_up = np.ones_like(y_test)
    y_prob_always_up = np.ones_like(y_test, dtype=float)
    always_up = direction_metrics(y_test, y_pred_always_up, y_prob_always_up)

    momentum_pred = (test_df["stock_return_1d"].values > 0).astype(int)
    momentum_prob = momentum_pred.astype(float)
    momentum = direction_metrics(y_test, momentum_pred, momentum_prob)

    x_tab_train, _, y_train = prepare_matrices(train_df)
    sentiment_train = x_tab_train[:, :3]
    sentiment_test = x_tab_test[:, :3]
    sent_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    sent_model.fit(sentiment_train, y_train)
    sent_pred = sent_model.predict(sentiment_test)
    sent_prob_up = sent_model.predict_proba(sentiment_test)[:, 1]
    sentiment_only = direction_metrics(y_test, sent_pred, sent_prob_up)

    metrics = {
        "test_sample_size": int(len(test_df)),
        "train_sample_size": int(len(train_df)),
        "temporal_split": {
            "train_end": data_cfg["price_train_end_date"],
            "test_start": data_cfg["price_test_start_date"],
        },
        "stage2_calibrated_fusion": stage2,
        "baselines": {
            "always_up": always_up,
            "momentum": momentum,
            "sentiment_only": sentiment_only,
        },
        "limitation": (
            "Predicts next-day close vs event-day close direction using headline text, "
            "FinBERT outputs, and same-day market context. Correlational, not causal."
        ),
    }

    write_json(eval_cfg["metrics_module3_path"], metrics)

    y_pred_labels = np.where(y_pred == 1, "Up", "Down")
    y_true_labels = np.where(y_test == 1, "Up", "Down")
    plot_confusion_matrix(
        y_true_labels,
        y_pred_labels,
        ["Down", "Up"],
        os.path.join(eval_cfg["figures_path"], "confusion_matrix_price_model.png"),
        title="Stage 2 Price Direction — Confusion Matrix (2020 test)",
    )
    plot_calibration_curve(
        y_test,
        y_prob_up,
        os.path.join(eval_cfg["figures_path"], "calibration_curve.png"),
    )

    logger.info(
        "Stage 2 test accuracy: %.4f, MCC: %.4f, Brier: %.4f",
        stage2["accuracy"],
        stage2["matthews_correlation_coefficient"],
        stage2["brier_score"],
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
