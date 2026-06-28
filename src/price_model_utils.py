"""Shared utilities for Stage 2 price model training and evaluation (v2)."""

from __future__ import annotations

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler

from price_constants import (
    ABLATION_CONFIGS,
    MARKET_FEATURES,
    SENTIMENT_FEATURES,
    TABULAR_FEATURES,
    cls_column_names,
)

logger = logging.getLogger(__name__)


def temporal_split_v2(
    df: pd.DataFrame,
    train_end: str,
    val_start: str,
    test_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    train = df[df["trading_date"] <= pd.Timestamp(train_end)]
    val = df[
        (df["trading_date"] >= pd.Timestamp(val_start))
        & (df["trading_date"] < pd.Timestamp(test_start))
    ]
    test = df[df["trading_date"] >= pd.Timestamp(test_start)]
    return train, val, test


def temporal_split(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    train = df[df["trading_date"] <= pd.Timestamp(train_end)]
    test = df[df["trading_date"] >= pd.Timestamp(test_start)]
    return train, test


def prepare_matrices(
    df: pd.DataFrame,
    ablation: str = "full_fusion",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature matrices for v2 ablation config."""
    cfg = ABLATION_CONFIGS[ablation]
    cls_cols = cls_column_names()
    y = (df["forward_direction"] == "Up").astype(int).values

    parts: list[np.ndarray] = []
    if cfg["use_sentiment"]:
        parts.append(df[SENTIMENT_FEATURES].astype(float).fillna(0.0).values)
    if cfg["use_market"]:
        parts.append(df[MARKET_FEATURES].astype(float).fillna(0.0).values)

    x_tab = np.hstack(parts) if parts else np.empty((len(df), 0))
    x_cls = df[cls_cols].astype(float).fillna(0.0).values if cfg["use_cls"] else None
    return x_tab, x_cls, y


def transform_with_pipeline_v2(
    pipeline: dict,
    x_tab: np.ndarray,
    x_cls: np.ndarray | None,
) -> np.ndarray:
    ablation = pipeline["ablation"]
    cfg = ABLATION_CONFIGS[ablation]
    parts: list[np.ndarray] = []

    if cfg["use_sentiment"] or cfg["use_market"]:
        parts.append(pipeline["scaler_tabular"].transform(x_tab))

    if cfg["use_cls"] and x_cls is not None:
        cls_pca = pipeline["pca"].transform(x_cls)
        parts.append(pipeline["scaler_cls"].transform(cls_pca))

    if not parts:
        raise ValueError("Empty feature matrix for ablation config.")
    return np.hstack(parts)


def find_optimal_threshold(y_true: np.ndarray, y_prob_up: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds and return (best_threshold, best_mcc)."""
    best_t, best_mcc = 0.5, -1.0
    for t in np.linspace(0.35, 0.65, 61):
        preds = (y_prob_up >= t).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_t = float(t)
    return best_t, best_mcc


def predict_with_threshold(y_prob_up: np.ndarray, threshold: float) -> np.ndarray:
    return (y_prob_up >= threshold).astype(int)


def build_lgbm_calibrated(config: dict, random_seed: int):
    import lightgbm as lgb

    price_cfg = config["models"]["price_direction_v2"]
    base = lgb.LGBMClassifier(
        n_estimators=price_cfg["lgbm_n_estimators"],
        learning_rate=price_cfg["lgbm_learning_rate"],
        num_leaves=price_cfg["lgbm_num_leaves"],
        class_weight="balanced",
        random_state=random_seed,
        verbose=-1,
    )
    return CalibratedClassifierCV(
        base,
        method=price_cfg["calibration_method"],
        cv=price_cfg["calibration_cv"],
    )
