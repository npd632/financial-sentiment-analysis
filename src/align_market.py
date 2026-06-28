#!/usr/bin/env python3
"""News filtering, price alignment, and market labels (README sections 4.2–4.4)."""

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import download_prices, load_prices
from preprocess import clean_text, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

NEWS_OUTPUT_COLS = ["headline", "stock", "date", "cleaned_text"]
ALIGNED_OUTPUT_COLS = [
    "headline",
    "stock",
    "news_datetime",
    "trading_date",
    "daily_return",
    "price_direction",
    "cleaned_text",
]
CHUNK_SIZE = 500_000
EASTERN = "America/New_York"


def to_trading_date(news_ts: pd.Timestamp) -> pd.Timestamp:
    """
    Convert a news timestamp to the NYSE trading date (US/Eastern calendar date).

    Saturday and Sunday map to the following Monday.
    """
    eastern = news_ts.tz_convert(EASTERN)
    trading_date = eastern.normalize()

    weekday = eastern.weekday()
    if weekday == 5:
        trading_date += pd.Timedelta(days=2)
    elif weekday == 6:
        trading_date += pd.Timedelta(days=1)

    return trading_date.tz_localize(None)


def align_news_with_prices(
    news_path: str,
    prices_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Join news headlines with daily prices and assign Up/Down labels (README 4.4).
    """
    if not os.path.exists(news_path):
        raise FileNotFoundError(f"News subset not found: {news_path}")
    if not os.path.exists(prices_path):
        raise FileNotFoundError(f"Price cache not found: {prices_path}")

    logger.info("Aligning news from %s with prices from %s", news_path, prices_path)

    news = pd.read_csv(news_path)
    initial_rows = len(news)
    news["news_datetime"] = pd.to_datetime(news["date"], utc=True)
    news["trading_date"] = news["news_datetime"].apply(to_trading_date)

    prices = load_prices(prices_path)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()

    merged = news.merge(
        prices,
        left_on=["stock", "trading_date"],
        right_on=["stock", "date"],
        how="inner",
    )
    missing_price = initial_rows - len(merged)
    if missing_price:
        logger.info("Dropped %d rows with missing price data", missing_price)

    merged["daily_return"] = (merged["Close"] - merged["Open"]) / merged["Open"]

    before_zero = len(merged)
    merged = merged[merged["daily_return"] != 0].copy()
    zero_return = before_zero - len(merged)
    if zero_return:
        logger.info("Dropped %d rows with zero daily return", zero_return)

    merged["price_direction"] = merged["daily_return"].apply(
        lambda r: "Up" if r > 0 else "Down"
    )
    merged["trading_date"] = merged["trading_date"].dt.strftime("%Y-%m-%d")
    merged["news_datetime"] = merged["news_datetime"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )

    output = merged[ALIGNED_OUTPUT_COLS]

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output.to_csv(output_path, index=False)
    logger.info(
        "Saved %d aligned rows to %s (from %d news rows)",
        len(output),
        output_path,
        initial_rows,
    )
    return output


def build_news_subset(
    input_path: str,
    output_path: str,
    tickers: list[str],
    start_date: str,
) -> pd.DataFrame:
    """
    Filter analyst ratings to the configured ticker list and date range,
    clean headlines, and write news_subset.csv (README section 4.2).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"News dataset not found: {input_path}")

    ticker_set = set(tickers)
    start_ts = pd.Timestamp(start_date, tz="UTC")
    logger.info(
        "Building news subset from %s (tickers=%d, start_date=%s)",
        input_path,
        len(ticker_set),
        start_date,
    )

    filtered_chunks: list[pd.DataFrame] = []
    total_rows = 0

    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)

        if "title" not in chunk.columns:
            raise ValueError(
                f"Required column 'title' not found. Available: {list(chunk.columns)}"
            )

        chunk = chunk.rename(columns={"title": "headline"})
        chunk = chunk.drop(columns=["Unnamed: 0"], errors="ignore")
        chunk["date"] = pd.to_datetime(chunk["date"], utc=True, errors="coerce")

        before = len(chunk)
        chunk = chunk[chunk["date"].notna()]
        chunk = chunk[chunk["date"] >= start_ts]
        chunk = chunk[chunk["stock"].isin(ticker_set)]
        dropped = before - len(chunk)
        if dropped:
            logger.debug("Chunk filtered out %d rows", dropped)

        if not chunk.empty:
            filtered_chunks.append(chunk)

    if not filtered_chunks:
        raise ValueError("No news rows matched the ticker and date filters.")

    df = pd.concat(filtered_chunks, ignore_index=True)
    logger.info("Read %d rows; %d rows after ticker/date filter", total_rows, len(df))

    df = df.dropna(subset=["headline", "stock"]).copy()
    df["cleaned_text"] = df["headline"].apply(clean_text)
    before_clean = len(df)
    df = df[df["cleaned_text"] != ""]
    if before_clean > len(df):
        logger.info("Dropped %d rows with empty cleaned headlines", before_clean - len(df))

    df["date"] = df["date"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df[NEWS_OUTPUT_COLS].to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)
    return df[NEWS_OUTPUT_COLS]


def main(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    data_cfg = config["data"]

    build_news_subset(
        input_path=data_cfg["news_path"],
        output_path=data_cfg["news_subset_path"],
        tickers=data_cfg["news_tickers"],
        start_date=data_cfg["news_start_date"],
    )

    download_prices(
        tickers=data_cfg["news_tickers"],
        start_date=data_cfg["price_start_date"],
        end_date=data_cfg["price_end_date"],
        output_path=data_cfg["prices_daily_path"],
    )

    align_news_with_prices(
        news_path=data_cfg["news_subset_path"],
        prices_path=data_cfg["prices_daily_path"],
        output_path=data_cfg["aligned_news_prices_path"],
    )


if __name__ == "__main__":
    main()
