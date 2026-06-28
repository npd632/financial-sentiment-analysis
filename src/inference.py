#!/usr/bin/env python3
"""Shared inference helpers for Stage 1 + Stage 2 price direction prediction."""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from data_loader import load_prices, load_spy
from finbert_inference import extract_cls_embeddings, predict_finbert_with_probs
from preprocess import clean_text
from price_constants import ABLATION_CONFIGS, EASTERN, MARKET_FEATURES
from price_features import compute_market_features_for_row, compute_price_features
from price_model_utils import predict_with_threshold, resolve_artifact_path, transform_with_pipeline


def load_best_model_info(config: dict) -> dict:
    path = os.path.join(config["models"]["price_direction"]["save_path"], "best_model.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Best model not found: {path}. Run train_price_model.py first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_price_pipeline(config: dict) -> dict:
    info = load_best_model_info(config)
    return joblib.load(resolve_artifact_path(info["pipeline_path"]))


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
    prices = compute_price_features(prices, spy)
    spy["date"] = pd.to_datetime(spy["date"]).dt.normalize()

    if news_datetime.tzinfo is None:
        news_datetime = news_datetime.tz_localize("UTC")
    hour_of_day = int(news_datetime.tz_convert(EASTERN).hour)

    return compute_market_features_for_row(
        ticker=ticker,
        trading_date=trading_date,
        prices=prices,
        spy=spy,
        hour_of_day=hour_of_day,
    )


def assemble_features(
    prob_negative: float,
    prob_neutral: float,
    prob_positive: float,
    market_features: dict,
    cls_embedding: np.ndarray | None,
    pipeline: dict,
) -> np.ndarray:
    ablation = pipeline["ablation"]
    cfg = ABLATION_CONFIGS[ablation]
    parts: list[list[float]] = []
    if cfg["use_sentiment"]:
        parts.append([prob_negative, prob_neutral, prob_positive])
    if cfg["use_market"]:
        parts.append([market_features[f] for f in MARKET_FEATURES])
    x_tab = np.array([np.concatenate(parts)], dtype=float) if parts else np.empty((1, 0))
    x_cls = cls_embedding.reshape(1, -1) if cfg["use_cls"] and cls_embedding is not None else None
    return transform_with_pipeline(pipeline, x_tab, x_cls)


def predict_price_direction(
    headline: str,
    ticker: str,
    trading_date,
    config: dict,
    pipeline: dict | None = None,
) -> dict:
    """
    Run Stage 1 + Stage 2 inference for one headline.

    Returns sentiment probs, predicted direction, P(Up), P(Down), confidence.
    """
    data_cfg = config["data"]
    finbert_cfg = config["models"]["finbert"]
    news_cfg = config["models"]["finbert_news"]

    if pipeline is None:
        pipeline = load_price_pipeline(config)

    cleaned = clean_text(headline)
    if not cleaned:
        raise ValueError("Headline is empty after cleaning.")

    finbert_dir = (
        news_cfg["save_path"]
        if pipeline.get("finbert_variant") == "news"
        else finbert_cfg["save_path"]
    )
    max_length = finbert_cfg["max_length"]

    probs_df = predict_finbert_with_probs(
        [cleaned],
        model_dir=finbert_dir,
        max_length=max_length,
        batch_size=1,
    )
    prob_row = probs_df.iloc[0]

    cls_embedding = None
    if ABLATION_CONFIGS[pipeline["ablation"]]["use_cls"]:
        cls_matrix = extract_cls_embeddings(
            [cleaned],
            model_dir=finbert_dir,
            max_length=max_length,
            batch_size=1,
        )
        cls_embedding = cls_matrix[0]

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
        cls_embedding=cls_embedding,
        pipeline=pipeline,
    )
    threshold = pipeline.get("optimal_threshold", 0.5)

    calibrator = pipeline["calibrator"]
    proba = calibrator.predict_proba(features)[0]
    p_down, p_up = float(proba[0]), float(proba[1])
    pred_id = int(predict_with_threshold(np.array([p_up]), threshold)[0])
    direction = "Up" if pred_id == 1 else "Down"

    return {
        "sentiment": prob_row["predicted_sentiment"],
        "prob_negative": float(prob_row["prob_negative"]),
        "prob_neutral": float(prob_row["prob_neutral"]),
        "prob_positive": float(prob_row["prob_positive"]),
        "predicted_direction": direction,
        "prob_up": p_up,
        "prob_down": p_down,
        "confidence": max(p_up, p_down),
        "cleaned_text": cleaned,
        "label_mode": "excess_1d",
        "optimal_threshold": threshold,
    }
