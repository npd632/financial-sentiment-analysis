#!/usr/bin/env python3
"""Build price-model dataset: forward labels, market features, FinBERT probs + CLS embeddings."""

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
from preprocess import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EASTERN = "America/New_York"
TABULAR_FEATURES = [
    "prob_negative",
    "prob_neutral",
    "prob_positive",
    "stock_return_1d",
    "stock_return_5d",
    "spy_return_1d",
    "volume_zscore_20d",
    "day_of_week",
    "hour_of_day",
]
CLS_PREFIX = "cls_"
NUM_CLS_DIM = 768


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


def compute_price_features(prices: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """Precompute per (stock, date) return and volume features."""
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices.sort_values(["stock", "date"])

    prices["stock_return_1d"] = prices.groupby("stock")["Close"].pct_change()
    prices["stock_return_5d"] = prices.groupby("stock")["Close"].pct_change(periods=5)

    vol_mean = prices.groupby("stock")["Volume"].transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    vol_std = prices.groupby("stock")["Volume"].transform(
        lambda s: s.rolling(20, min_periods=5).std()
    )
    prices["volume_zscore_20d"] = (prices["Volume"] - vol_mean) / vol_std.replace(0, np.nan)

    prices["close_next"] = prices.groupby("stock")["Close"].shift(-1)
    prices["forward_return"] = (prices["close_next"] - prices["Close"]) / prices["Close"]

    spy = spy.copy()
    spy["date"] = pd.to_datetime(spy["date"]).dt.normalize()
    spy = spy.sort_values("date")
    spy["spy_return_1d"] = spy["Close"].pct_change()
    spy_feats = spy[["date", "spy_return_1d"]]

    prices = prices.merge(spy_feats, on="date", how="left")
    return prices


def build_price_dataset(config: dict) -> pd.DataFrame:
    """Build the Stage 2 training dataset from aligned news and price data."""
    data_cfg = config["data"]
    finbert_cfg = config["models"]["finbert"]

    aligned_path = data_cfg["aligned_news_prices_path"]
    prices_path = data_cfg["prices_daily_path"]
    spy_path = data_cfg["spy_prices_path"]
    output_path = data_cfg["price_model_dataset_path"]
    flat_threshold = float(data_cfg["forward_flat_threshold"])

    if not os.path.exists(aligned_path):
        raise FileNotFoundError(f"Aligned data not found: {aligned_path}. Run align_market.py first.")

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
        "stock_return_1d",
        "stock_return_5d",
        "spy_return_1d",
        "volume_zscore_20d",
    ]
    df = df.merge(
        price_feats[merge_cols],
        left_on=["stock", "trading_date"],
        right_on=["stock", "date"],
        how="inner",
    )

    before = len(df)
    df = df[df["forward_return"].notna()].copy()
    df = df[df["forward_return"].abs() >= flat_threshold].copy()
    df["forward_direction"] = np.where(df["forward_return"] > 0, "Up", "Down")

    df["day_of_week"] = df["trading_date"].dt.dayofweek
    df["hour_of_day"] = (
        df["news_datetime"].dt.tz_convert(EASTERN).dt.hour.astype(int)
    )

    for col in ["stock_return_1d", "stock_return_5d", "spy_return_1d", "volume_zscore_20d"]:
        df[col] = df[col].fillna(0.0)

    logger.info(
        "Dropped %d rows (missing or flat forward returns); %d remain",
        before - len(df),
        len(df),
    )

    model_dir = finbert_cfg["save_path"]
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(
            f"FinBERT model not found at {model_dir}. Run train_finbert.py first."
        )

    logger.info("Running FinBERT inference for Stage 1 probabilities...")
    probs = predict_finbert_with_probs(
        list(df["cleaned_text"]),
        model_dir=model_dir,
        max_length=finbert_cfg["max_length"],
        batch_size=finbert_cfg["batch_size"],
    )
    df = pd.concat([df.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)

    logger.info("Extracting FinBERT CLS embeddings...")
    cls_matrix = extract_cls_embeddings(
        list(df["cleaned_text"]),
        model_dir=model_dir,
        max_length=finbert_cfg["max_length"],
        batch_size=finbert_cfg["batch_size"],
    )
    cls_cols = [f"{CLS_PREFIX}{i}" for i in range(NUM_CLS_DIM)]
    cls_df = pd.DataFrame(cls_matrix, columns=cls_cols)
    df = pd.concat([df.reset_index(drop=True), cls_df], axis=1)

    keep_cols = [
        "headline",
        "stock",
        "news_datetime",
        "trading_date",
        "forward_return",
        "forward_direction",
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


def main(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    build_price_dataset(config)


if __name__ == "__main__":
    main()
