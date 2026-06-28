#!/usr/bin/env python3
"""Train Stage 2 calibrated price-direction classifier (v1 LogReg or v2 LightGBM)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from price_constants import (
    ABLATION_CONFIGS,
    MARKET_FEATURES,
    SENTIMENT_FEATURES,
    V1_TABULAR_FEATURES,
    cls_column_names,
)
from price_model_utils import (
    build_lgbm_calibrated,
    find_optimal_threshold,
    prepare_matrices,
    temporal_split,
    temporal_split_v2,
    transform_with_pipeline_v2,
)
from preprocess import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_matrices_v1(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cls_cols = cls_column_names()
    y = (df["forward_direction"] == "Up").astype(int).values
    x_tab = df[V1_TABULAR_FEATURES].astype(float).fillna(0.0).values
    x_cls = df[cls_cols].astype(float).fillna(0.0).values
    return x_tab, x_cls, y


def train_price_model_v1(config: dict) -> None:
    data_cfg = config["data"]
    price_cfg = config["models"]["price_direction"]
    dataset_path = data_cfg["price_model_dataset_path"]

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)
    train_df, test_df = temporal_split(
        df,
        train_end=data_cfg["price_train_end_date"],
        test_start=data_cfg["price_test_start_date"],
    )
    logger.info("v1 temporal split — train: %d, test: %d", len(train_df), len(test_df))

    x_tab_train, x_cls_train, y_train = prepare_matrices_v1(train_df)
    x_tab_test, x_cls_test, y_test = prepare_matrices_v1(test_df)

    scaler_tabular = StandardScaler()
    x_tab_train_s = scaler_tabular.fit_transform(x_tab_train)
    x_tab_test_s = scaler_tabular.transform(x_tab_test)

    pca = PCA(n_components=price_cfg["pca_components"], random_state=data_cfg["random_seed"])
    x_cls_train_p = pca.fit_transform(x_cls_train)
    x_cls_test_p = pca.transform(x_cls_test)

    scaler_cls = StandardScaler()
    x_cls_train_s = scaler_cls.fit_transform(x_cls_train_p)
    x_cls_test_s = scaler_cls.transform(x_cls_test_p)

    x_train = np.hstack([x_tab_train_s, x_cls_train_s])
    x_test = np.hstack([x_tab_test_s, x_cls_test_s])

    base = LogisticRegression(
        C=price_cfg["logistic_c"],
        max_iter=1000,
        class_weight="balanced",
        random_state=data_cfg["random_seed"],
    )
    calibrator = CalibratedClassifierCV(
        base,
        method=price_cfg["calibration_method"],
        cv=price_cfg["calibration_cv"],
    )
    calibrator.fit(x_train, y_train)
    logger.info(
        "v1 train acc: %.4f | test acc: %.4f",
        calibrator.score(x_train, y_train),
        calibrator.score(x_test, y_test),
    )

    save_dir = price_cfg["save_path"]
    os.makedirs(save_dir, exist_ok=True)
    pipeline = {
        "version": "v1",
        "scaler_tabular": scaler_tabular,
        "pca": pca,
        "scaler_cls": scaler_cls,
        "calibrator": calibrator,
        "tabular_features": V1_TABULAR_FEATURES,
        "optimal_threshold": 0.5,
        "ablation": "full_fusion",
    }
    joblib.dump(pipeline, os.path.join(save_dir, "pipeline.pkl"))
    logger.info("Saved v1 pipeline to %s", save_dir)


def train_single_ablation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
    ablation: str,
    finbert_variant: str,
) -> tuple[dict, float]:
    data_cfg = config["data"]
    price_cfg = config["models"]["price_direction_v2"]
    seed = data_cfg["random_seed"]

    x_tab_train, x_cls_train, y_train = prepare_matrices(train_df, ablation=ablation)
    x_tab_val, x_cls_val, y_val = prepare_matrices(val_df, ablation=ablation)

    cfg = ABLATION_CONFIGS[ablation]
    pipeline: dict = {
        "version": "v2",
        "ablation": ablation,
        "finbert_variant": finbert_variant,
        "label_mode": "excess_1d",
        "tabular_features": (
            (SENTIMENT_FEATURES if cfg["use_sentiment"] else [])
            + (MARKET_FEATURES if cfg["use_market"] else [])
        ),
    }

    if cfg["use_sentiment"] or cfg["use_market"]:
        scaler_tabular = StandardScaler()
        x_tab_train_s = scaler_tabular.fit_transform(x_tab_train)
        x_tab_val_s = scaler_tabular.transform(x_tab_val)
        pipeline["scaler_tabular"] = scaler_tabular
    else:
        x_tab_train_s = x_tab_train
        x_tab_val_s = x_tab_val

    if cfg["use_cls"]:
        pca = PCA(n_components=price_cfg["pca_components"], random_state=seed)
        x_cls_train_p = pca.fit_transform(x_cls_train)
        x_cls_val_p = pca.transform(x_cls_val)
        scaler_cls = StandardScaler()
        x_cls_train_s = scaler_cls.fit_transform(x_cls_train_p)
        x_cls_val_s = scaler_cls.transform(x_cls_val_p)
        pipeline["pca"] = pca
        pipeline["scaler_cls"] = scaler_cls
        x_train = np.hstack([x_tab_train_s, x_cls_train_s])
        x_val = np.hstack([x_tab_val_s, x_cls_val_s])
    else:
        x_train = x_tab_train_s
        x_val = x_tab_val_s

    calibrator = build_lgbm_calibrated(config, seed)
    calibrator.fit(x_train, y_train)
    pipeline["calibrator"] = calibrator

    y_prob_val = calibrator.predict_proba(x_val)[:, 1]
    optimal_t, val_mcc = find_optimal_threshold(y_val, y_prob_val)
    pipeline["optimal_threshold"] = optimal_t
    pipeline["val_mcc"] = val_mcc

    return pipeline, val_mcc


def train_price_model_v2(config: dict) -> None:
    data_cfg = config["data"]
    price_cfg = config["models"]["price_direction_v2"]
    save_root = price_cfg["save_path"]

    variants = {
        "phrasebank": data_cfg["price_model_dataset_phrasebank_path"],
        "news": data_cfg["price_model_dataset_news_path"],
    }

    best_overall: dict | None = None
    best_mcc = -2.0
    results_summary: dict = {}

    for finbert_variant, dataset_path in variants.items():
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}. "
                f"Run build_price_dataset.py --version v2 --finbert-variant {finbert_variant}"
            )

        df = pd.read_parquet(dataset_path)
        train_df, val_df, test_df = temporal_split_v2(
            df,
            train_end=data_cfg["train_end_date_v2"],
            val_start=data_cfg["validation_start_date"],
            test_start=data_cfg["price_test_start_date"],
        )
        logger.info(
            "%s split — train: %d, val: %d, test: %d",
            finbert_variant,
            len(train_df),
            len(val_df),
            len(test_df),
        )

        for ablation in ABLATION_CONFIGS:
            name = f"{finbert_variant}_{ablation}"
            logger.info("Training v2 %s...", name)
            pipeline, val_mcc = train_single_ablation(
                train_df, val_df, config, ablation, finbert_variant
            )

            out_dir = os.path.join(save_root, name)
            os.makedirs(out_dir, exist_ok=True)
            joblib.dump(pipeline, os.path.join(out_dir, "pipeline.pkl"))
            meta = {
                "finbert_variant": finbert_variant,
                "ablation": ablation,
                "val_mcc": val_mcc,
                "optimal_threshold": pipeline["optimal_threshold"],
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
            }
            with open(os.path.join(out_dir, "ablation_config.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)

            results_summary[name] = val_mcc
            if val_mcc > best_mcc:
                best_mcc = val_mcc
                best_overall = {
                    "model_id": name,
                    "pipeline_path": os.path.join(out_dir, "pipeline.pkl").replace("\\", "/"),
                    "finbert_variant": finbert_variant,
                    "ablation": ablation,
                    "val_mcc": val_mcc,
                    "optimal_threshold": pipeline["optimal_threshold"],
                }

    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, "best_model.json"), "w", encoding="utf-8") as f:
        json.dump(best_overall, f, indent=4)

    logger.info("v2 training complete. Best model: %s (val MCC=%.4f)", best_overall["model_id"], best_mcc)
    logger.info("Ablation val MCC summary: %s", results_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 price direction model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.version == "v2":
        train_price_model_v2(config)
    else:
        train_price_model_v1(config)


if __name__ == "__main__":
    main()
