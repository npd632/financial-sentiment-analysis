#!/usr/bin/env python3
"""PhraseBank preprocessing (README section 4.1)."""

import logging
import os
import re

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TEXT_COL = "Sentence"
LABEL_COL = "Sentiment"
OUTPUT_COLS = [TEXT_COL, LABEL_COL, "cleaned_text"]


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration from the project root."""
    logger.info("Loading config from %s", config_path)
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(text: object) -> str:
    """
    Clean financial text for baseline and FinBERT pipelines.

    Steps: strip HTML/URLs, lowercase, keep alphabetic characters only,
    collapse whitespace.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_phrasebank(
    input_path: str,
    output_path: str,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """Load PhraseBank, clean text, and write processed CSV."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"PhraseBank dataset not found: {input_path}")

    logger.info("Loading PhraseBank from %s", input_path)
    df = pd.read_csv(input_path)
    initial_rows = len(df)

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Required columns {text_col!r} and {label_col!r} not found. "
            f"Available: {list(df.columns)}"
        )

    df = df.dropna(subset=[text_col, label_col]).copy()
    df["cleaned_text"] = df[text_col].apply(clean_text)
    df = df[df["cleaned_text"] != ""]

    dropped = initial_rows - len(df)
    if dropped:
        logger.info("Dropped %d rows with null or empty text/labels", dropped)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df[OUTPUT_COLS].to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)
    return df[OUTPUT_COLS]


def main(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    data_cfg = config["data"]

    preprocess_phrasebank(
        input_path=data_cfg["phrasebank_path"],
        output_path=data_cfg["phrasebank_clean_path"],
    )


if __name__ == "__main__":
    main()
