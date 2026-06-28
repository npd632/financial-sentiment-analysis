"""Shared constants for Stage 2 price-direction modeling."""

SENTIMENT_FEATURES = ["prob_negative", "prob_neutral", "prob_positive"]

V1_MARKET_FEATURES = [
    "stock_return_1d",
    "stock_return_5d",
    "spy_return_1d",
    "volume_zscore_20d",
    "day_of_week",
    "hour_of_day",
]

MARKET_FEATURES = [
    "stock_return_1d",
    "stock_return_5d",
    "spy_return_1d",
    "stock_excess_return_1d",
    "stock_excess_return_5d",
    "realized_vol_20d",
    "intraday_return",
    "gap_return",
    "volume_zscore_20d",
    "day_of_week",
    "hour_of_day",
]

V1_TABULAR_FEATURES = SENTIMENT_FEATURES + V1_MARKET_FEATURES
TABULAR_FEATURES = SENTIMENT_FEATURES + MARKET_FEATURES

CLS_PREFIX = "cls_"
NUM_CLS_DIM = 768

ABLATION_CONFIGS = {
    "sentiment_only": {"use_sentiment": True, "use_market": False, "use_cls": False},
    "market_only": {"use_sentiment": False, "use_market": True, "use_cls": False},
    "sentiment_market": {"use_sentiment": True, "use_market": True, "use_cls": False},
    "full_fusion": {"use_sentiment": True, "use_market": True, "use_cls": True},
}


def cls_column_names() -> list[str]:
    return [f"{CLS_PREFIX}{i}" for i in range(NUM_CLS_DIM)]
