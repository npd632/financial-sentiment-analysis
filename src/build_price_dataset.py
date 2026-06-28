#!/usr/bin/env python3
"""Build price-model dataset (v1 raw return or v2 excess return + company filter)."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import download_spy, load_prices, load_spy
from finbert_inference import predict_finbert_with_probs
from headline_filter import apply_company_filter
from preprocess import load_config
from price_constants import (
    CLS_PREFIX,
    MARKET_FEATURES,
    NUM_CLS_DIM,
    SENTIMENT_FEATURES,
    TABULAR_FEATURES,
    cls_column_names,
)
from price_features import compute_price_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EASTERN = "America/New_York"


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


def _resolve_v2_settings(config: dict, finbert_variant: str) -> tuple[dict, str, str]:
    data_cfg = config["data"]
    finbert_cfg = config["models"]["finbert"]
    news_cfg = config["models"]["finbert_news"]

    if finbert_variant == "phrasebank":
        model_dir = finbert_cfg["save_path"]
        output_path = data_cfg["price_model_dataset_phrasebank_path"]
    elif finbert_variant == "news":
        model_dir = news_cfg["save_path"]
        output_path = data_cfg["price_model_dataset_news_path"]
    else:
        raise ValueError(f"Unknown finbert variant: {finbert_variant}")

    settings = {
        "flat_threshold": float(data_cfg["forward_flat_threshold_v2"]),
        "use_excess": data_cfg.get("forward_label_mode_v2", "excess_1d") == "excess_1d",
        "company_filter": bool(data_cfg.get("company_filter_enabled_v2", True)),
        "aliases_path": data_cfg["ticker_aliases_path"],
    }
    return settings, model_dir, output_path


def build_labeled_frame(config: dict, version: str) -> pd.DataFrame:
    """Load aligned news, merge price features, apply filters, assign labels."""
    data_cfg = config["data"]
    aligned_path = data_cfg["aligned_news_prices_path"]
    prices_path = data_cfg["prices_daily_path"]
    spy_path = data_cfg["spy_prices_path"]

    if version == "v2":
        flat_threshold = float(data_cfg["forward_flat_threshold_v2"])
        use_excess = True
        company_filter = bool(data_cfg.get("company_filter_enabled_v2", True))
    else:
        flat_threshold = float(data_cfg["forward_flat_threshold"])
        use_excess = False
        company_filter = False

    if not os.path.exists(spy_path):
        download_spy(
            start_date=data_cfg["price_start_date"],
            end_date=data_cfg["spy_end_date"],
            output_path=spy_path,
        )

    logger.info("Loading aligned news from %s", aligned_path)
    df = pd.read_csv(aligned_path)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.normalize()
    df["news_datetime"] = pd.to_datetime(df["news_datetime"], utc=True)

    prices = load_prices(prices_path)
    spy = load_spy(spy_path)
    price_feats = compute_price_features(prices, spy)

    merge_cols = [
        "stock",
        "date",
        "forward_return",
        "excess_forward_return",
        "stock_return_1d",
        "stock_return_5d",
        "spy_return_1d",
        "stock_excess_return_1d",
        "stock_excess_return_5d",
        "realized_vol_20d",
        "intraday_return",
        "gap_return",
        "volume_zscore_20d",
    ]
    df = df.merge(
        price_feats[merge_cols],
        left_on=["stock", "trading_date"],
        right_on=["stock", "date"],
        how="inner",
    )

    if company_filter:
        before_filter = len(df)
        df, _ = apply_company_filter(
            df, data_cfg["ticker_aliases_path"], text_col="cleaned_text", ticker_col="stock"
        )
        df = df[df["headline_relevant"]].copy()
        logger.info(
            "Company filter: kept %d / %d rows (%.1f%%)",
            len(df),
            before_filter,
            100.0 * len(df) / max(before_filter, 1),
        )
    else:
        df["headline_relevant"] = True

    label_col = "excess_forward_return" if use_excess else "forward_return"
    before = len(df)
    df = df[df[label_col].notna()].copy()
    df = df[df[label_col].abs() >= flat_threshold].copy()
    if not use_excess:
        df["excess_forward_return"] = df.get("excess_forward_return", 0.0).fillna(0.0)

    df["forward_direction"] = np.where(df[label_col] > 0, "Up", "Down")
    df["day_of_week"] = df["trading_date"].dt.dayofweek
    df["hour_of_day"] = df["news_datetime"].dt.tz_convert(EASTERN).dt.hour.astype(int)

    fill_cols = [c for c in MARKET_FEATURES if c in df.columns]
    for col in fill_cols:
        df[col] = df[col].fillna(0.0)

    logger.info(
        "Dropped %d rows (missing or flat %s); %d remain",
        before - len(df),
        label_col,
        len(df),
    )
    return df


def attach_finbert_features(
    df: pd.DataFrame,
    model_dir: str,
    max_length: int,
    batch_size: int,
) -> pd.DataFrame:
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(f"FinBERT model not found at {model_dir}.")

    logger.info("Running FinBERT inference from %s", model_dir)
    probs = predict_finbert_with_probs(
        list(df["cleaned_text"]),
        model_dir=model_dir,
        max_length=max_length,
        batch_size=batch_size,
    )
    df = pd.concat([df.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)

    logger.info("Extracting CLS embeddings...")
    cls_matrix = extract_cls_embeddings(
        list(df["cleaned_text"]),
        model_dir=model_dir,
        max_length=max_length,
        batch_size=batch_size,
    )
    cls_cols = cls_column_names()
    cls_df = pd.DataFrame(cls_matrix, columns=cls_cols)
    return pd.concat([df.reset_index(drop=True), cls_df], axis=1)


def build_price_dataset(
    config: dict,
    version: str = "v1",
    finbert_variant: str = "phrasebank",
) -> pd.DataFrame:
    """Build Stage 2 training dataset."""
    finbert_cfg = config["models"]["finbert"]
    news_cfg = config["models"]["finbert_news"]

    if version == "v2":
        _, model_dir, output_path = _resolve_v2_settings(config, finbert_variant)
        max_length = news_cfg.get("max_length", finbert_cfg["max_length"])
        batch_size = news_cfg.get("batch_size", finbert_cfg["batch_size"])
    else:
        model_dir = finbert_cfg["save_path"]
        output_path = config["data"]["price_model_dataset_path"]
        max_length = finbert_cfg["max_length"]
        batch_size = finbert_cfg["batch_size"]

    df = build_labeled_frame(config, version=version)
    df = attach_finbert_features(df, model_dir, max_length, batch_size)

    cls_cols = cls_column_names()
    keep_cols = [
        "headline",
        "stock",
        "news_datetime",
        "trading_date",
        "forward_return",
        "excess_forward_return",
        "forward_direction",
        "headline_relevant",
        "cleaned_text",
        *TABULAR_FEATURES,
        *cls_cols,
    ]
    df = df[keep_cols]
    df["news_datetime"] = df["news_datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    df["trading_date"] = df["trading_date"].dt.strftime("%Y-%m-%d")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build price direction dataset.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    parser.add_argument(
        "--finbert-variant",
        choices=["phrasebank", "news"],
        default="phrasebank",
        help="Required for v2: which Stage-1 checkpoint to use for probs/CLS",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    build_price_dataset(
        config,
        version=args.version,
        finbert_variant=args.finbert_variant,
    )


if __name__ == "__main__":
    main()
