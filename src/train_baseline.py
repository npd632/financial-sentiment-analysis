#!/usr/bin/env python3
"""Baseline sentiment models: TF-IDF + Naive Bayes + SVM (README section 5.1)."""

import logging
import os
import pickle
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TEXT_COL = "cleaned_text"
LABEL_COL = "Sentiment"


def train_baseline(config: dict) -> None:
    """Train Naive Bayes and SVM on preprocessed PhraseBank and save artifacts."""
    data_cfg = config["data"]
    baseline_cfg = config["models"]["baseline"]

    input_path = data_cfg["phrasebank_clean_path"]
    train_split = data_cfg["train_split"]
    random_seed = data_cfg["random_seed"]
    save_dir = baseline_cfg["save_path"]

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
    logger.info("Training on %d samples", len(df))

    logger.info(
        "Splitting dataset: train=%.0f%%, test=%.0f%%, random_state=%d",
        train_split * 100,
        (1 - train_split) * 100,
        random_seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        df[TEXT_COL],
        df[LABEL_COL],
        test_size=(1.0 - train_split),
        random_state=random_seed,
        stratify=df[LABEL_COL],
    )

    max_features = baseline_cfg["tfidf_max_features"]
    logger.info("Fitting TF-IDF vectorizer (max_features=%d)", max_features)
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    models = {
        "naive_bayes": MultinomialNB(alpha=baseline_cfg["naive_bayes_alpha"]),
        "svm": SVC(
            kernel=baseline_cfg["svm_kernel"],
            probability=True,
            random_state=random_seed,
        ),
    }

    os.makedirs(save_dir, exist_ok=True)

    for name, model in models.items():
        logger.info("Training %s...", name)
        model.fit(x_train_tfidf, y_train)
        y_pred = model.predict(x_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        _, _, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        logger.info(
            "%s — accuracy: %.4f, macro F1: %.4f, weighted F1: %.4f",
            name,
            accuracy,
            f1_macro,
            f1_weighted,
        )

        model_path = os.path.join(save_dir, f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved model to %s", model_path)

    vectorizer_path = os.path.join(save_dir, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Saved vectorizer to %s", vectorizer_path)


def main(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    train_baseline(config)
    logger.info("Baseline training completed.")


if __name__ == "__main__":
    main()
