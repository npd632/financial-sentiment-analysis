#!/usr/bin/env python3
"""Download and load market price data via yfinance (README section 4.3)."""

import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]
OUTPUT_COLS = ["stock", "date", *PRICE_COLS]


def download_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    output_path: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Download daily OHLCV for each ticker and cache to parquet.

    Returns the combined dataframe, successful tickers, and failed tickers.
    """
    # yfinance end date is exclusive; add one day for an inclusive end_date.
    end_exclusive = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )

    frames: list[pd.DataFrame] = []
    succeeded: list[str] = []
    failed: list[str] = []

    logger.info(
        "Downloading prices for %d tickers (%s to %s)",
        len(tickers),
        start_date,
        end_date,
    )

    for ticker in tickers:
        try:
            history = yf.Ticker(ticker).history(
                start=start_date,
                end=end_exclusive,
                auto_adjust=False,
            )
        except Exception as exc:
            logger.warning("Failed to download %s: %s", ticker, exc)
            failed.append(ticker)
            continue

        if history.empty:
            logger.warning("No price data returned for %s; excluding ticker", ticker)
            failed.append(ticker)
            continue

        df = history.reset_index()
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "date"})

        missing_cols = [col for col in PRICE_COLS if col not in df.columns]
        if missing_cols:
            logger.warning(
                "Missing columns %s for %s; excluding ticker",
                missing_cols,
                ticker,
            )
            failed.append(ticker)
            continue

        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None).dt.normalize()
        df["stock"] = ticker
        frames.append(df[OUTPUT_COLS])
        succeeded.append(ticker)
        logger.info("Downloaded %d rows for %s", len(df), ticker)

    if not frames:
        raise RuntimeError("Price download failed for all tickers.")

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.drop_duplicates(subset=["stock", "date"]).sort_values(
        ["stock", "date"]
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    prices.to_parquet(output_path, index=False)
    logger.info(
        "Saved %d rows for %d tickers to %s",
        len(prices),
        len(succeeded),
        output_path,
    )

    if failed:
        logger.warning("Excluded tickers with failed downloads: %s", ", ".join(failed))

    return prices, succeeded, failed


def load_prices(path: str) -> pd.DataFrame:
    """Load cached daily price data from parquet."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Price cache not found: {path}")
    return pd.read_parquet(path)


def main(config_path: str = "config.yaml") -> None:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocess import load_config

    config = load_config(config_path)
    data_cfg = config["data"]

    download_prices(
        tickers=data_cfg["news_tickers"],
        start_date=data_cfg["price_start_date"],
        end_date=data_cfg["price_end_date"],
        output_path=data_cfg["prices_daily_path"],
    )


if __name__ == "__main__":
    main()
