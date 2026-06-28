"""Shared constants for Stage 2 price-direction modeling."""

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


def cls_column_names() -> list[str]:
    return [f"{CLS_PREFIX}{i}" for i in range(NUM_CLS_DIM)]
