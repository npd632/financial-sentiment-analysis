#!/usr/bin/env python3
"""Shared inference helpers for Stage 1 + Stage 2 price direction prediction."""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from build_price_dataset import extract_cls_embeddings
from data_loader import load_prices, load_spy
from finbert_inference import predict_finbert_with_probs
from preprocess import clean_text

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


def load_price_pipeline(model_dir: str) -> dict:
    """Load saved Stage 2 artifacts."""
    path = os.path.join(model_dir, "pipeline.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Price direction pipeline not found: {path}. Run train_price_model.py first."
        )
    return joblib.load(path)


def compute_tabular_features_for_row(
    ticker: str,
    trading_date: pd.Timestamp,
    news_datetime: pd.Timestamp,
    prices_path: str,
    spy_path: str,
) -> dict:
    """Compute market tabular features for a single ticker/date (excludes FinBERT probs)."""
    prices = load_prices(prices_path)
    spy = load_spy(spy_path)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    spy["date"] = pd.to_datetime(spy["date"]).dt.normalize()
    trading_date = pd.to_datetime(trading_date).normalize()

    stock_hist = prices[prices["stock"] == ticker].sort_values("date")
    if stock_hist.empty:
        raise ValueError(f"No price history for ticker {ticker}")

    idx = stock_hist.index[stock_hist["date"] == trading_date]
    if len(idx) == 0:
        raise ValueError(f"No price row for {ticker} on {trading_date.date()}")

    pos = stock_hist.index.get_loc(idx[0])
    row = stock_hist.iloc[pos]
    prev = stock_hist.iloc[pos - 1] if pos > 0 else None
    prev5 = stock_hist.iloc[pos - 5] if pos >= 5 else None

    stock_return_1d = (
        (row["Close"] - prev["Close"]) / prev["Close"] if prev is not None else 0.0
    )
    stock_return_5d = (
        (row["Close"] - prev5["Close"]) / prev5["Close"] if prev5 is not None else 0.0
    )

    window = stock_hist.iloc[max(0, pos - 19) : pos + 1]["Volume"]
    vol_mean = window.mean()
    vol_std = window.std()
    volume_zscore_20d = (
        (row["Volume"] - vol_mean) / vol_std if vol_std and vol_std > 0 else 0.0
    )

    spy_row = spy[spy["date"] == trading_date]
    spy_prev = spy[spy["date"] < trading_date].tail(1)
    if not spy_row.empty and not spy_prev.empty:
        spy_return_1d = (spy_row.iloc[0]["Close"] - spy_prev.iloc[0]["Close"]) / spy_prev.iloc[
            0
        ]["Close"]
    else:
        spy_return_1d = 0.0

    if news_datetime.tzinfo is None:
        news_datetime = news_datetime.tz_localize("UTC")
    hour_of_day = int(news_datetime.tz_convert(EASTERN).hour)

    return {
        "stock_return_1d": float(stock_return_1d),
        "stock_return_5d": float(stock_return_5d),
        "spy_return_1d": float(spy_return_1d),
        "volume_zscore_20d": float(volume_zscore_20d),
        "day_of_week": int(trading_date.dayofweek),
        "hour_of_day": hour_of_day,
    }


def assemble_features(
    prob_negative: float,
    prob_neutral: float,
    prob_positive: float,
    market_features: dict,
    cls_embedding: np.ndarray,
    pipeline: dict,
) -> np.ndarray:
    """Apply fitted scalers/PCA and return model input matrix (1, n_features)."""
    tabular = np.array(
        [
            [
                prob_negative,
                prob_neutral,
                prob_positive,
                market_features["stock_return_1d"],
                market_features["stock_return_5d"],
                market_features["spy_return_1d"],
                market_features["volume_zscore_20d"],
                market_features["day_of_week"],
                market_features["hour_of_day"],
            ]
        ],
        dtype=float,
    )
    tabular_scaled = pipeline["scaler_tabular"].transform(tabular)
    cls_pca = pipeline["pca"].transform(cls_embedding.reshape(1, -1))
    cls_scaled = pipeline["scaler_cls"].transform(cls_pca)
    return np.hstack([tabular_scaled, cls_scaled])


def predict_price_direction(
    headline: str,
    ticker: str,
    trading_date,
    config: dict,
    pipeline: dict | None = None,
    finbert_model=None,
    finbert_tokenizer=None,
    finbert_device=None,
) -> dict:
    """
    Run Stage 1 + Stage 2 inference for one headline.

    Returns sentiment probs, predicted direction, P(Up), P(Down), confidence.
    """
    data_cfg = config["data"]
    finbert_cfg = config["models"]["finbert"]
    price_cfg = config["models"]["price_direction"]

    if pipeline is None:
        pipeline = load_price_pipeline(price_cfg["save_path"])

    cleaned = clean_text(headline)
    if not cleaned:
        raise ValueError("Headline is empty after cleaning.")

    probs_df = predict_finbert_with_probs(
        [cleaned],
        model_dir=finbert_cfg["save_path"],
        max_length=finbert_cfg["max_length"],
        batch_size=1,
    )
    prob_row = probs_df.iloc[0]

    cls_matrix = extract_cls_embeddings(
        [cleaned],
        model_dir=finbert_cfg["save_path"],
        max_length=finbert_cfg["max_length"],
        batch_size=1,
    )

    trading_ts = pd.to_datetime(trading_date)
    news_dt = pd.Timestamp(trading_ts).tz_localize("UTC")

    market = compute_tabular_features_for_row(
        ticker=ticker,
        trading_date=trading_ts,
        news_datetime=news_dt,
        prices_path=data_cfg["prices_daily_path"],
        spy_path=data_cfg["spy_prices_path"],
    )

    features = assemble_features(
        prob_negative=prob_row["prob_negative"],
        prob_neutral=prob_row["prob_neutral"],
        prob_positive=prob_row["prob_positive"],
        market_features=market,
        cls_embedding=cls_matrix[0],
        pipeline=pipeline,
    )

    calibrator = pipeline["calibrator"]
    pred_id = int(calibrator.predict(features)[0])
    proba = calibrator.predict_proba(features)[0]
    p_down, p_up = float(proba[0]), float(proba[1])
    direction = "Up" if pred_id == 1 else "Down"
    confidence = max(p_up, p_down)

    return {
        "sentiment": prob_row["predicted_sentiment"],
        "prob_negative": float(prob_row["prob_negative"]),
        "prob_neutral": float(prob_row["prob_neutral"]),
        "prob_positive": float(prob_row["prob_positive"]),
        "predicted_direction": direction,
        "prob_up": p_up,
        "prob_down": p_down,
        "confidence": confidence,
        "cleaned_text": cleaned,
    }


def load_feature_metadata(model_dir: str) -> dict:
    path = os.path.join(model_dir, "feature_columns.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
