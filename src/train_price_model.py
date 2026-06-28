#!/usr/bin/env python3
"""Train Stage 2 calibrated price-direction classifier."""

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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from price_constants import CLS_PREFIX, NUM_CLS_DIM, TABULAR_FEATURES, cls_column_names
from preprocess import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


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


def prepare_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cls_cols = cls_column_names()
    y = (df["forward_direction"] == "Up").astype(int).values
    x_tab = df[TABULAR_FEATURES].astype(float).fillna(0.0).values
    x_cls = df[cls_cols].astype(float).fillna(0.0).values
    return x_tab, x_cls, y


def train_price_model(config: dict) -> None:
    data_cfg = config["data"]
    price_cfg = config["models"]["price_direction"]

    dataset_path = data_cfg["price_model_dataset_path"]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Price model dataset not found: {dataset_path}. "
            "Run `python src/build_price_dataset.py` first."
        )

    logger.info("Loading %s", dataset_path)
    df = pd.read_parquet(dataset_path)
    train_df, test_df = temporal_split(
        df,
        train_end=data_cfg["price_train_end_date"],
        test_start=data_cfg["price_test_start_date"],
    )
    logger.info("Temporal split — train: %d, test: %d", len(train_df), len(test_df))

    x_tab_train, x_cls_train, y_train = prepare_matrices(train_df)
    x_tab_test, x_cls_test, y_test = prepare_matrices(test_df)

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
    logger.info("Training calibrated logistic regression...")
    calibrator.fit(x_train, y_train)

    train_acc = calibrator.score(x_train, y_train)
    test_acc = calibrator.score(x_test, y_test)
    logger.info("Train accuracy: %.4f | Test accuracy: %.4f", train_acc, test_acc)

    save_dir = price_cfg["save_path"]
    os.makedirs(save_dir, exist_ok=True)

    pipeline = {
        "scaler_tabular": scaler_tabular,
        "pca": pca,
        "scaler_cls": scaler_cls,
        "calibrator": calibrator,
        "tabular_features": TABULAR_FEATURES,
        "pca_components": price_cfg["pca_components"],
        "label_up": 1,
        "label_down": 0,
    }
    joblib.dump(pipeline, os.path.join(save_dir, "pipeline.pkl"))

    metadata = {
        "tabular_features": TABULAR_FEATURES,
        "cls_columns": cls_column_names(),
        "pca_components": price_cfg["pca_components"],
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_end_date": data_cfg["price_train_end_date"],
        "test_start_date": data_cfg["price_test_start_date"],
    }
    with open(os.path.join(save_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info("Saved pipeline to %s", save_dir)


def main(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    train_price_model(config)


if __name__ == "__main__":
    main()
