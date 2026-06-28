"""Shared price feature computation for dataset build and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_price_features(prices: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """Precompute per (stock, date) return, excess, and volume features."""
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices.sort_values(["stock", "date"])

    prices["prev_close"] = prices.groupby("stock")["Close"].shift(1)
    prices["stock_return_1d"] = prices.groupby("stock")["Close"].pct_change()
    prices["stock_return_5d"] = prices.groupby("stock")["Close"].pct_change(periods=5)
    prices["intraday_return"] = (prices["Close"] - prices["Open"]) / prices["Open"]
    prices["gap_return"] = (prices["Open"] - prices["prev_close"]) / prices["prev_close"]
    prices["realized_vol_20d"] = prices.groupby("stock")["stock_return_1d"].transform(
        lambda s: s.rolling(20, min_periods=5).std()
    )

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
    spy["spy_return_5d"] = spy["Close"].pct_change(periods=5)
    spy["spy_close_next"] = spy["Close"].shift(-1)
    spy["spy_forward_return"] = (spy["spy_close_next"] - spy["Close"]) / spy["Close"]
    spy_feats = spy[
        ["date", "spy_return_1d", "spy_return_5d", "spy_forward_return"]
    ]

    prices = prices.merge(spy_feats, on="date", how="left")
    prices["stock_excess_return_1d"] = prices["stock_return_1d"] - prices["spy_return_1d"]
    prices["stock_excess_return_5d"] = prices["stock_return_5d"] - prices["spy_return_5d"]
    prices["excess_forward_return"] = prices["forward_return"] - prices["spy_forward_return"]

    return prices


def compute_market_features_for_row(
    ticker: str,
    trading_date: pd.Timestamp,
    prices: pd.DataFrame,
    spy: pd.DataFrame,
    hour_of_day: int,
) -> dict:
    """Compute market tabular features for one ticker/date (no sentiment probs)."""
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
    intraday_return = float(row.get("intraday_return", 0.0) or 0.0)
    gap_return = float(row.get("gap_return", 0.0) or 0.0)
    realized_vol_20d = float(row.get("realized_vol_20d", 0.0) or 0.0)

    window = stock_hist.iloc[max(0, pos - 19) : pos + 1]["Volume"]
    vol_mean = window.mean()
    vol_std = window.std()
    volume_zscore_20d = (
        (row["Volume"] - vol_mean) / vol_std if vol_std and vol_std > 0 else 0.0
    )

    spy_row = spy[spy["date"] == trading_date]
    spy_prev = spy[spy["date"] < trading_date].tail(1)
    spy_prev5 = spy[spy["date"] <= trading_date].tail(6).head(1)
    if not spy_row.empty and not spy_prev.empty:
        spy_return_1d = (spy_row.iloc[0]["Close"] - spy_prev.iloc[0]["Close"]) / spy_prev.iloc[
            0
        ]["Close"]
    else:
        spy_return_1d = 0.0
    if not spy_row.empty and not spy_prev5.empty:
        spy_return_5d = (spy_row.iloc[0]["Close"] - spy_prev5.iloc[0]["Close"]) / spy_prev5.iloc[
            0
        ]["Close"]
    else:
        spy_return_5d = 0.0

    return {
        "stock_return_1d": float(stock_return_1d),
        "stock_return_5d": float(stock_return_5d),
        "spy_return_1d": float(spy_return_1d),
        "stock_excess_return_1d": float(stock_return_1d - spy_return_1d),
        "stock_excess_return_5d": float(stock_return_5d - spy_return_5d),
        "realized_vol_20d": realized_vol_20d,
        "intraday_return": intraday_return,
        "gap_return": gap_return,
        "volume_zscore_20d": float(volume_zscore_20d),
        "day_of_week": int(trading_date.dayofweek),
        "hour_of_day": hour_of_day,
    }
