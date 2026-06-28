#!/usr/bin/env python3
"""Streamlit demo: Stage 1 sentiment + Stage 2 next-day excess return vs SPY."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inference import load_best_model_info, load_price_pipeline, predict_price_direction
from preprocess import load_config

SENTIMENT_EMOJI = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}


@st.cache_resource
def load_pipeline_cached(config_dict: dict):
    return load_price_pipeline(config_dict)


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
    confidence_threshold = price_cfg["confidence_threshold"]

    best_model_path = ROOT / price_cfg["save_path"] / "best_model.json"
    dataset_path = ROOT / data_cfg["price_model_dataset_phrasebank_path"]

    st.title("Financial Sentiment & Price Outlook")
    st.caption(
        "Two-stage pipeline: FinBERT sentiment (Stage 1) + calibrated **next-day excess return vs SPY** "
        "(Stage 2)."
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

        if best_model_path.exists():
            best = load_best_model_info(
                {
                    **config,
                    "models": {
                        **config["models"],
                        "price_direction": {
                            **price_cfg,
                            "save_path": str(ROOT / price_cfg["save_path"]),
                        },
                    },
                }
            )
            st.caption(f"Stage 2 model: `{best['model_id']}`")

        if dataset_path.exists():
            dataset = load_price_dataset(str(dataset_path))
            rows = dataset[
                (dataset["stock"] == ticker) & (dataset["trading_date"] == trading_date)
            ]
            if not rows.empty:
                st.success(f"{len(rows)} cached headline(s) for {ticker} on {trading_date}")
                sample = rows.iloc[0]
                st.metric("Actual excess return (next day)", f"{sample['excess_forward_return']:.4f}")
                st.metric("Actual excess direction", sample["forward_direction"])
            else:
                st.info("No cached rows for this ticker/date in the price dataset.")
        else:
            st.warning("Run dataset build to enable ground-truth lookup.")

    st.subheader("Headline analysis")
    headline = st.text_area(
        "Enter a financial news headline",
        height=120,
        placeholder="e.g. Nvidia shares surge after company reports record datacenter revenue growth",
    )

    if st.button("Analyze", type="primary"):
        if not headline.strip():
            st.error("Please enter a headline.")
            return

        if not best_model_path.exists():
            st.error(
                "Price model not found. Run:\n"
                "`python src/train_finbert_news.py`\n"
                "`python src/build_price_dataset.py --finbert-variant phrasebank`\n"
                "`python src/build_price_dataset.py --finbert-variant news`\n"
                "`python src/train_price_model.py`"
            )
            return

        runtime_config = {
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
                "finbert_news": {
                    **config["models"]["finbert_news"],
                    "save_path": str(ROOT / config["models"]["finbert_news"]["save_path"]),
                },
                "price_direction": {
                    **price_cfg,
                    "save_path": str(ROOT / price_cfg["save_path"]),
                },
            },
        }

        with st.spinner("Running Stage 1 + Stage 2 inference..."):
            pipeline = load_pipeline_cached(runtime_config)
            result = predict_price_direction(
                headline=headline,
                ticker=ticker,
                trading_date=trading_date,
                config=runtime_config,
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
            st.markdown("#### Stage 2 — Next-day excess return vs SPY")
            direction = result["predicted_direction"]
            st.markdown(f"**Predicted excess direction: {direction}**")
            st.metric("P(Up vs SPY)", f"{result['prob_up']:.2%}")
            st.metric("P(Down vs SPY)", f"{result['prob_down']:.2%}")
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
                    f"Ground truth (cached): excess direction **{row['forward_direction']}**, "
                    f"excess return **{row['excess_forward_return']:.4f}**"
                )

        with st.expander("Details"):
            st.write("Cleaned text:", result["cleaned_text"])
            st.json(
                {
                    "label_mode": result["label_mode"],
                    "optimal_threshold": round(result["optimal_threshold"], 4),
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
