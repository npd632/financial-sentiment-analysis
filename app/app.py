#!/usr/bin/env python3
"""Streamlit demo: Stage 1 sentiment + Stage 2 next-day price direction with confidence."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inference import load_price_pipeline, predict_price_direction
from preprocess import load_config

SENTIMENT_EMOJI = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}


@st.cache_resource
def load_pipeline_cached(model_dir: str):
    return load_price_pipeline(model_dir)


@st.cache_data
def load_price_dataset(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    return df


def lookup_forward_row(
    dataset: pd.DataFrame,
    ticker: str,
    trading_date,
    headline: str | None = None,
) -> pd.Series | None:
    """Find a row in the price model dataset for verification."""
    mask = (dataset["stock"] == ticker) & (dataset["trading_date"] == trading_date)
    if headline:
        mask &= dataset["headline"] == headline
    matches = dataset[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def main() -> None:
    st.set_page_config(
        page_title="Financial Sentiment & Price Outlook",
        page_icon="📈",
        layout="wide",
    )

    config = load_config(str(ROOT / "config.yaml"))
    data_cfg = config["data"]
    price_cfg = config["models"]["price_direction"]

    pipeline_path = ROOT / price_cfg["save_path"] / "pipeline.pkl"
    dataset_path = ROOT / data_cfg["price_model_dataset_path"]
    confidence_threshold = price_cfg["confidence_threshold"]

    st.title("Financial Sentiment & Price Outlook")
    st.caption(
        "Two-stage system: FinBERT sentiment (Stage 1) + calibrated next-day "
        "price direction model (Stage 2)."
    )

    tickers = data_cfg["news_tickers"]
    price_start = pd.to_datetime(data_cfg["price_start_date"]).date()
    price_end = pd.to_datetime(data_cfg["price_end_date"]).date()

    with st.sidebar:
        st.header("Context")
        ticker = st.selectbox("Ticker (required for price outlook)", tickers)
        trading_date = st.date_input(
            "Event trading date",
            value=price_end,
            min_value=price_start,
            max_value=price_end,
        )

        if dataset_path.exists():
            dataset = load_price_dataset(str(dataset_path))
            rows = dataset[
                (dataset["stock"] == ticker) & (dataset["trading_date"] == trading_date)
            ]
            if not rows.empty:
                st.success(f"{len(rows)} cached headline(s) for {ticker} on {trading_date}")
                sample = rows.iloc[0]
                st.metric("Actual next-day return", f"{sample['forward_return']:.4f}")
                st.metric("Actual forward direction", sample["forward_direction"])
            else:
                st.info("No cached rows for this ticker/date in the evaluation dataset.")
        else:
            st.warning("Run `python src/build_price_dataset.py` to enable ground-truth lookup.")

    st.subheader("Headline analysis")
    headline = st.text_area(
        "Enter a financial news headline",
        height=120,
        placeholder="e.g. Company beats earnings expectations and raises full-year guidance",
    )

    if st.button("Analyze", type="primary"):
        if not headline.strip():
            st.error("Please enter a headline.")
            return

        if not pipeline_path.exists():
            st.error(
                "Price direction model not found. Run:\n"
                "`python src/build_price_dataset.py`\n"
                "`python src/train_price_model.py`"
            )
            return

        with st.spinner("Running Stage 1 + Stage 2 inference..."):
            pipeline = load_pipeline_cached(str(ROOT / price_cfg["save_path"]))
            result = predict_price_direction(
                headline=headline,
                ticker=ticker,
                trading_date=trading_date,
                config={
                    **config,
                    "data": {
                        **data_cfg,
                        "prices_daily_path": str(ROOT / data_cfg["prices_daily_path"]),
                        "spy_prices_path": str(ROOT / data_cfg["spy_prices_path"]),
                    },
                    "models": {
                        **config["models"],
                        "finbert": {
                            **config["models"]["finbert"],
                            "save_path": str(ROOT / config["models"]["finbert"]["save_path"]),
                        },
                    },
                },
                pipeline=pipeline,
            )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Stage 1 — Sentiment")
            emoji = SENTIMENT_EMOJI.get(result["sentiment"], "")
            st.markdown(f"**{emoji} {result['sentiment'].title()}**")
            prob_df = pd.DataFrame(
                {
                    "class": ["negative", "neutral", "positive"],
                    "probability": [
                        result["prob_negative"],
                        result["prob_neutral"],
                        result["prob_positive"],
                    ],
                }
            ).set_index("class")
            st.bar_chart(prob_df)

        with col2:
            st.markdown("#### Stage 2 — Next-day price outlook")
            direction = result["predicted_direction"]
            st.markdown(f"**Predicted direction: {direction}**")
            st.metric("P(Up)", f"{result['prob_up']:.2%}")
            st.metric("P(Down)", f"{result['prob_down']:.2%}")
            st.metric("Confidence", f"{result['confidence']:.2%}")

            if result["confidence"] < confidence_threshold:
                st.warning(
                    f"Low confidence (< {confidence_threshold:.0%}). "
                    "Treat this prediction with caution."
                )

        if dataset_path.exists():
            row = lookup_forward_row(
                load_price_dataset(str(dataset_path)),
                ticker,
                trading_date,
                headline=headline.strip(),
            )
            if row is not None:
                st.info(
                    f"Ground truth (cached): forward direction **{row['forward_direction']}**, "
                    f"return **{row['forward_return']:.4f}**"
                )

        with st.expander("Details"):
            st.write("Cleaned text:", result["cleaned_text"])
            st.json(
                {
                    "sentiment": result["sentiment"],
                    "prob_negative": round(result["prob_negative"], 4),
                    "prob_neutral": round(result["prob_neutral"], 4),
                    "prob_positive": round(result["prob_positive"], 4),
                    "predicted_direction": result["predicted_direction"],
                    "prob_up": round(result["prob_up"], 4),
                    "prob_down": round(result["prob_down"], 4),
                    "confidence": round(result["confidence"], 4),
                }
            )


if __name__ == "__main__":
    main()
