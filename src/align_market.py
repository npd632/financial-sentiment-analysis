#!/usr/bin/env python3
"""News filtering, price alignment, and market labels (README sections 4.2–4.4)."""

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

NEWS_OUTPUT_COLS = ["headline", "stock", "date", "cleaned_text"]
CHUNK_SIZE = 500_000


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


if __name__ == "__main__":
    main()
