#!/usr/bin/env python3
"""Build Stage 2 training dataset (excess return labels + company filter + FinBERT features)."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import download_spy, load_prices, load_spy
from finbert_inference import extract_cls_embeddings, predict_finbert_with_probs
from headline_filter import apply_company_filter
from preprocess import load_config
from price_constants import (
    EASTERN,
    MARKET_FEATURES,
    TABULAR_FEATURES,
    cls_column_names,
)
from price_features import compute_price_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_finbert_settings(config: dict, finbert_variant: str) -> tuple[str, str]:
    data_cfg = config["data"]
    finbert_cfg = config["models"]["finbert"]
    news_cfg = config["models"]["finbert_news"]

    if finbert_variant == "phrasebank":
        return finbert_cfg["save_path"], data_cfg["price_model_dataset_phrasebank_path"]
    if finbert_variant == "news":
        return news_cfg["save_path"], data_cfg["price_model_dataset_news_path"]
    raise ValueError(f"Unknown finbert variant: {finbert_variant}")


def build_labeled_frame(config: dict) -> pd.DataFrame:
    """Load aligned news, merge price features, apply filters, assign excess-return labels."""
    data_cfg = config["data"]
    flat_threshold = float(data_cfg["forward_flat_threshold"])
    company_filter = bool(data_cfg.get("company_filter_enabled", True))

    if not os.path.exists(data_cfg["spy_prices_path"]):
        download_spy(
            start_date=data_cfg["price_start_date"],
            end_date=data_cfg["spy_end_date"],
            output_path=data_cfg["spy_prices_path"],
        )

    logger.info("Loading aligned news from %s", data_cfg["aligned_news_prices_path"])
    df = pd.read_csv(data_cfg["aligned_news_prices_path"])
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.normalize()
    df["news_datetime"] = pd.to_datetime(df["news_datetime"], utc=True)

    prices = load_prices(data_cfg["prices_daily_path"])
    spy = load_spy(data_cfg["spy_prices_path"])
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

    label_col = "excess_forward_return"
    before = len(df)
    df = df[df[label_col].notna()].copy()
    df = df[df[label_col].abs() >= flat_threshold].copy()

    df["forward_direction"] = np.where(df[label_col] > 0, "Up", "Down")
    df["day_of_week"] = df["trading_date"].dt.dayofweek
    df["hour_of_day"] = df["news_datetime"].dt.tz_convert(EASTERN).dt.hour.astype(int)

    for col in MARKET_FEATURES:
        if col in df.columns:
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
    cls_df = pd.DataFrame(cls_matrix, columns=cls_column_names())
    return pd.concat([df.reset_index(drop=True), cls_df], axis=1)


def build_price_dataset(config: dict, finbert_variant: str = "phrasebank") -> pd.DataFrame:
    """Build Stage 2 training dataset for one FinBERT variant."""
    finbert_cfg = config["models"]["finbert"]
    news_cfg = config["models"]["finbert_news"]
    model_dir, output_path = _resolve_finbert_settings(config, finbert_variant)
    max_length = news_cfg.get("max_length", finbert_cfg["max_length"])
    batch_size = news_cfg.get("batch_size", finbert_cfg["batch_size"])

    df = build_labeled_frame(config)
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
    parser.add_argument(
        "--finbert-variant",
        choices=["phrasebank", "news"],
        default="phrasebank",
        help="Which Stage-1 checkpoint to use for probs/CLS",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    build_price_dataset(config, finbert_variant=args.finbert_variant)


if __name__ == "__main__":
    main()
