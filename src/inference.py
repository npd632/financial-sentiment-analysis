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
from price_constants import MARKET_FEATURES, SENTIMENT_FEATURES
from price_features import compute_market_features_for_row, compute_price_features
from price_model_utils import predict_with_threshold, transform_with_pipeline_v2

EASTERN = "America/New_York"


def load_best_model_info(config: dict) -> dict:
    path = os.path.join(config["models"]["price_direction_v2"]["save_path"], "best_model.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"v2 best model not found: {path}. Run train_price_model.py --version v2 first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_price_pipeline_v2(config: dict) -> dict:
    info = load_best_model_info(config)
    path = info["pipeline_path"]
    if not os.path.isabs(path) and not os.path.exists(path):
        # Resolve relative to cwd (project root when running streamlit/scripts)
        alt = os.path.join(os.getcwd(), path)
        if os.path.exists(alt):
            path = alt
    return joblib.load(path)


def load_price_pipeline(model_dir: str) -> dict:
    """Load saved Stage 2 artifacts (v1 flat path)."""
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


def assemble_features_v1(
    prob_negative: float,
    prob_neutral: float,
    prob_positive: float,
    market_features: dict,
    cls_embedding: np.ndarray,
    pipeline: dict,
) -> np.ndarray:
    """v1 LogReg pipeline feature assembly."""
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


def assemble_features_v2(
    prob_negative: float,
    prob_neutral: float,
    prob_positive: float,
    market_features: dict,
    cls_embedding: np.ndarray | None,
    pipeline: dict,
) -> np.ndarray:
    from price_constants import ABLATION_CONFIGS

    ablation = pipeline["ablation"]
    cfg = ABLATION_CONFIGS[ablation]
    parts: list[list[float]] = []
    if cfg["use_sentiment"]:
        parts.append([prob_negative, prob_neutral, prob_positive])
    if cfg["use_market"]:
        parts.append([market_features[f] for f in MARKET_FEATURES])
    x_tab = np.array([np.concatenate(parts)], dtype=float) if parts else np.empty((1, 0))
    x_cls = cls_embedding.reshape(1, -1) if cfg["use_cls"] and cls_embedding is not None else None
    return transform_with_pipeline_v2(pipeline, x_tab, x_cls)


def predict_price_direction(
    headline: str,
    ticker: str,
    trading_date,
    config: dict,
    pipeline: dict | None = None,
    use_v2: bool = True,
) -> dict:
    """
    Run Stage 1 + Stage 2 inference for one headline.

    Returns sentiment probs, predicted direction, P(Up), P(Down), confidence.
    """
    data_cfg = config["data"]
    finbert_cfg = config["models"]["finbert"]
    news_cfg = config["models"]["finbert_news"]

    if pipeline is None:
        if use_v2:
            pipeline = load_price_pipeline_v2(config)
        else:
            pipeline = load_price_pipeline(config["models"]["price_direction"]["save_path"])

    cleaned = clean_text(headline)
    if not cleaned:
        raise ValueError("Headline is empty after cleaning.")

    if use_v2 and pipeline.get("finbert_variant") == "news":
        finbert_dir = news_cfg["save_path"]
    else:
        finbert_dir = finbert_cfg["save_path"]

    max_length = finbert_cfg["max_length"]
    probs_df = predict_finbert_with_probs(
        [cleaned],
        model_dir=finbert_dir,
        max_length=max_length,
        batch_size=1,
    )
    prob_row = probs_df.iloc[0]

    cls_embedding = None
    if pipeline.get("version") == "v2":
        from price_constants import ABLATION_CONFIGS

        if ABLATION_CONFIGS[pipeline["ablation"]]["use_cls"]:
            cls_matrix = extract_cls_embeddings(
                [cleaned],
                model_dir=finbert_dir,
                max_length=max_length,
                batch_size=1,
            )
            cls_embedding = cls_matrix[0]
    else:
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

    if pipeline.get("version") == "v2":
        features = assemble_features_v2(
            prob_negative=prob_row["prob_negative"],
            prob_neutral=prob_row["prob_neutral"],
            prob_positive=prob_row["prob_positive"],
            market_features=market,
            cls_embedding=cls_embedding,
            pipeline=pipeline,
        )
        threshold = pipeline.get("optimal_threshold", 0.5)
    else:
        features = assemble_features_v1(
            prob_negative=prob_row["prob_negative"],
            prob_neutral=prob_row["prob_neutral"],
            prob_positive=prob_row["prob_positive"],
            market_features=market,
            cls_embedding=cls_embedding,
            pipeline=pipeline,
        )
        threshold = 0.5

    calibrator = pipeline["calibrator"]
    proba = calibrator.predict_proba(features)[0]
    p_down, p_up = float(proba[0]), float(proba[1])
    pred_id = int(predict_with_threshold(np.array([p_up]), threshold)[0])
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
        "label_mode": "excess_1d" if use_v2 else "raw_1d",
        "optimal_threshold": threshold,
    }


def load_feature_metadata(model_dir: str) -> dict:
    path = os.path.join(model_dir, "feature_columns.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
