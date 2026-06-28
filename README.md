# Financial Sentiment Analysis & Market Trend Prediction

## 1. Project Overview

This project is a **two-stage financial NLP + ML system**:

1. **Stage 1 (sentiment):** Classify headline tone (positive / negative / neutral) using FinBERT fine-tuned on PhraseBank.
2. **Stage 2 (price direction):** Predict **most likely next-day excess return vs SPY** (Up/Down) with **calibrated probabilities**, using Stage 1 outputs plus market context features.

**Research question (v2):** Given a **company-relevant** headline, ticker, and event-day market state, can we predict whether the stock's **next-day return will beat SPY** — and how confident should we be?

The advanced sentiment model is **FinBERT** (`yiyanghkust/finbert-tone`), fine-tuned on PhraseBank. **v2** adds a **news-adapted FinBERT** (`models/finbert_news_adapted/`) and Stage 2 **LightGBM** with ablation search and MCC-optimal threshold.

Two local datasets serve distinct roles:

| Dataset | Path | Rows | Role |
|---------|------|------|------|
| Financial PhraseBank | `data/raw/PhraseBank/data.csv` | 5,842 | Supervised sentiment training and model comparison |
| Daily Financial News (processed analyst ratings) | `data/raw/DailyFinancialNews/analyst_ratings_processed.csv` | ~1.4M | Build price-model dataset and evaluate next-day excess return prediction |

---

## 2. System Architecture

The pipeline runs in **5 phases**:

| Phase | Name | Primary outputs |
|-------|------|-----------------|
| 1 | Data pipeline | `phrasebank_clean.csv`, `news_subset.csv`, `prices_daily.parquet`, `spy_prices.parquet`, `aligned_news_prices.csv`, **`price_model_dataset.parquet`** |
| 2 | Modeling | `models/baseline/*`, `models/finbert_finetuned/`, **`models/price_direction/`** |
| 3 | Sentiment evaluation | `metrics_module1.json`, `metrics_module2.json`, confusion matrices |
| 4 | **Price direction evaluation** | `metrics_module3.json`, `confusion_matrix_price_model.png`, `calibration_curve.png` |
| 5 | Deployment | Streamlit demo: sentiment + **next-day direction + confidence** |

```mermaid
flowchart LR
  subgraph phase1 [Phase1_DataPipeline]
    PB[PhraseBank]
    News[news_subset]
    YF[yfinance_OHLCV]
    SPY[SPY_context]
    PB --> CleanPB[phrasebank_clean.csv]
    News --> Aligned[aligned_news_prices.csv]
    YF --> Prices[prices_daily.parquet]
    SPY --> Prices
    Aligned --> PriceDS[price_model_dataset.parquet]
    Prices --> PriceDS
  end
  subgraph phase2 [Phase2_Modeling]
    CleanPB --> FinBERT[FinBERT_sentiment]
    FinBERT --> Embed[CLS_embeddings]
    PriceDS --> Embed
    Embed --> Stage2[LogReg_calibrated]
  end
  subgraph phase3 [Phase3_SentimentEval]
    FinBERT --> M1M2[metrics_module1_2]
  end
  subgraph phase4 [Phase4_PriceDirectionEval]
    Stage2 --> M3[metrics_module3.json]
    Stage2 --> CalPlot[calibration_curve.png]
  end
  subgraph phase5 [Phase5_Demo]
    FinBERT --> Streamlit[Streamlit]
    Stage2 --> Streamlit
  end
```

---

## 3. Project Structure

```text
financial-sentiment-analysis/
│
├── data/
│   ├── raw/
│   │   ├── PhraseBank/data.csv
│   │   └── DailyFinancialNews/analyst_ratings_processed.csv
│   ├── processed/
│   │   ├── phrasebank_clean.csv
│   │   ├── news_subset.csv
│   │   ├── aligned_news_prices.csv
│   │   └── price_model_dataset.parquet
│   └── external/
│       ├── prices_daily.parquet
│       └── spy_prices.parquet
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_baseline_testing.ipynb
│
├── src/
│   ├── data_loader.py          # yfinance download (tickers + SPY)
│   ├── preprocess.py           # PhraseBank cleaning
│   ├── align_market.py         # news filtering, same-day price alignment
│   ├── build_price_dataset.py  # forward labels, features, FinBERT probs + CLS
│   ├── headline_filter.py      # company-relevance filter (v2)
│   ├── price_features.py       # shared market feature computation
│   ├── price_model_utils.py    # v2 splits, ablations, threshold tuning
│   ├── finbert_inference.py    # shared FinBERT inference helpers
│   ├── price_constants.py      # Stage 2 feature column definitions
│   ├── train_baseline.py       # TF-IDF + Naive Bayes + SVM
│   ├── train_finbert.py        # FinBERT fine-tuning (Stage 1A)
│   ├── train_finbert_news.py   # news multi-task FinBERT (Stage 1B)
│   ├── train_price_model.py    # Stage 2 v1 LogReg / v2 LightGBM
│   ├── inference.py            # Stage 1 + Stage 2 demo inference
│   └── evaluate.py             # sentiment and price-direction metrics
│
├── models/
│   ├── baseline/
│   ├── finbert_finetuned/
│   └── price_direction/
│       ├── pipeline.pkl
│       └── feature_columns.json
│
├── evaluation/
│   ├── metrics_module1.json
│   ├── metrics_module2.json
│   ├── metrics_module3.json
│   └── figures/
│       ├── confusion_matrix_price_model.png
│       └── calibration_curve.png
│
├── app/app.py
├── config.yaml
├── requirements.txt
└── README.md
```

All hyperparameters and paths are defined in `config.yaml` (single source of truth).

---

## 4. Phase 1 — Data Pipeline

### 4.1 PhraseBank preprocessing (`src/preprocess.py`)

**Input:** `data/raw/PhraseBank/data.csv`

**Text cleaning** (identical for baseline and FinBERT):

1. Strip HTML tags and URLs
2. Convert to lowercase
3. Remove non-alphabetic characters
4. Collapse whitespace
5. Drop rows with null or empty text

Stop words are **not** removed at this stage. English stop words are applied only inside the TF-IDF vectorizer during baseline training.

**Output:** `data/processed/phrasebank_clean.csv` with columns `Sentence`, `Sentiment`, `cleaned_text`.

### 4.2 News subset (`src/align_market.py`)

**Input:** `data/raw/DailyFinancialNews/analyst_ratings_processed.csv`

**Filter criteria:**

- `stock` must be in the top 19 tickers by headline count (BBRY excluded — no yfinance data)
- `date >= 2018-01-01`

**Top 19 tickers (fixed):**

`MRK`, `MS`, `MU`, `NVDA`, `QQQ`, `M`, `EBAY`, `NFLX`, `GILD`, `VZ`, `DAL`, `JNJ`, `QCOM`, `BABA`, `KO`, `ORCL`, `FDX`, `HD`, `WFC`

**Processing steps:**

1. Rename column `title` → `headline`
2. Parse `date` to UTC timestamps; drop rows with unparseable dates
3. Apply the same text cleaning as PhraseBank; store result in `cleaned_text`

**Output:** `data/processed/news_subset.csv`

### 4.3 Price data (`src/data_loader.py` + `src/align_market.py`)

**Source:** `yfinance` daily OHLCV (Open, High, Low, Close, Volume)

**Scope:** 19 tickers above, date range **2018-01-01 to 2020-06-11**

**Cache:** `data/external/prices_daily.parquet` (one row per ticker/date)

Tickers that fail to download are logged and excluded from alignment.

### 4.4 Trading-day alignment (same-day labels for legacy dataset)

1. Convert each news timestamp to a US/Eastern calendar date
2. If the date falls on Saturday or Sunday, shift to the **next Monday** (NYSE next-trading-day rule)
3. Join news rows with price data on `(stock, trading_date)`
4. Compute daily return: `(Close - Open) / Open`
5. Assign same-day price label: `Up` if return > 0, `Down` if return < 0; drop flat/missing rows

**Output:** `data/processed/aligned_news_prices.csv`

### 4.5 SPY market context (`src/data_loader.py`)

- Download **SPY** daily OHLCV for **2018-01-01 to 2020-06-12** (extra day for last forward label)
- Cache: `data/external/spy_prices.parquet`
- Used only as **market context**, not as a prediction ticker

### 4.6 Price model dataset (`src/build_price_dataset.py`)

**Input:**

- `data/processed/aligned_news_prices.csv`
- `data/external/prices_daily.parquet`
- `data/external/spy_prices.parquet`
- Fine-tuned FinBERT at `models/finbert_finetuned/`

**Forward return label (per row):**

For each row with event trading date `t` and ticker `stock`:

1. Look up `Close[t]` and next trading day `Close[t+1]` from `prices_daily.parquet`
2. `forward_return = (Close[t+1] - Close[t]) / Close[t]`
3. Drop rows where `forward_return == 0` or `|forward_return| < 0.001` (0.1% flat band)
4. `forward_direction`: **Up** if `forward_return > 0`, **Down** if `< 0`

**Tabular features (known at end of day `t`):**

| Feature | Definition |
|---------|------------|
| `prob_negative`, `prob_neutral`, `prob_positive` | FinBERT softmax on `cleaned_text` |
| `stock_return_1d` | `(Close[t] - Close[t-1]) / Close[t-1]` |
| `stock_return_5d` | 5-day cumulative close-to-close return ending at `t` |
| `spy_return_1d` | SPY same formula on date `t` |
| `volume_zscore_20d` | Z-score of volume vs 20-day rolling mean for `stock` |
| `day_of_week` | 0–4 (Mon–Fri) from `trading_date` |
| `hour_of_day` | Hour from `news_datetime` (US/Eastern), 0–23 |

**Text embedding feature:**

- FinBERT **`[CLS]`** hidden state (768-dim) per `cleaned_text`, batch size 16
- Stored as `cls_0` … `cls_767` columns; **PCA (768 → 32)** is fit later in `train_price_model.py` on **train split only**

**Output:** `data/processed/price_model_dataset.parquet` (~16k rows after filtering)

### 4.7 Pipeline v2 extensions

**Company filter:** [`src/headline_filter.py`](src/headline_filter.py) keeps headlines mentioning the ticker or alias from [`config/ticker_aliases.yaml`](config/ticker_aliases.yaml) (~47% of aligned rows retained).

**Primary label (v2):** `excess_forward_return = stock_forward_return - spy_forward_return`; flat band **0.3%** on `|excess|`.

**Expanded features:** `stock_excess_return_1d/5d`, `realized_vol_20d`, `intraday_return`, `gap_return`.

**Dual datasets:**

| Variant | FinBERT checkpoint | Output |
|---------|------------------|--------|
| phrasebank | `models/finbert_finetuned/` | `price_model_dataset_v2_phrasebank.parquet` |
| news | `models/finbert_news_adapted/` | `price_model_dataset_v2_news.parquet` |

Build: `python src/build_price_dataset.py --version v2 --finbert-variant {phrasebank|news}` (~6.3k rows each after filter).

---

## 5. Phase 2 — Modeling

### 5.1 Baseline models (`src/train_baseline.py`)

**Input:** `data/processed/phrasebank_clean.csv`

**Features:** `TfidfVectorizer(max_features=5000, stop_words="english")`

**Models:** Multinomial Naive Bayes (`alpha=1.0`), Linear SVM (`kernel="linear"`, `probability=True`)

**Train/test split:** 80/20 stratified random split (`random_state=42`).

**Saved artifacts:** `models/baseline/naive_bayes.pkl`, `svm.pkl`, `tfidf_vectorizer.pkl`

### 5.2 FinBERT — Stage 1 (`src/train_finbert.py`)

**Pretrained model:** `yiyanghkust/finbert-tone`

**Label mapping:** `{neutral: 0, positive: 1, negative: 2}`

| Hyperparameter | Value |
|----------------|-------|
| Batch size | 16 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Max length | 128 |
| Best checkpoint metric | `macro_f1` |

**Saved artifacts:** `models/finbert_finetuned/` (model weights + tokenizer)

### 5.3 Stage 2 — Price direction model (`src/train_price_model.py`)

**Input:** `data/processed/price_model_dataset.parquet`

**Temporal split (fixed):**

| Set | `trading_date` range |
|-----|----------------------|
| Train | 2018-01-01 – 2019-12-31 |
| Test | 2020-01-01 – 2020-06-11 |

**Feature matrix (41 features):**

1. Stage 1 probs (3)
2. Tabular market features (6)
3. PCA-reduced CLS embedding (32)

**Base estimator:** `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`

**Calibration:** `CalibratedClassifierCV(estimator, method="isotonic", cv=3)` fit on **train only**

**Saved artifacts:**

- `models/price_direction/pipeline.pkl` — `StandardScaler` (tabular + PCA features) + `PCA(32)` + calibrator
- `models/price_direction/feature_columns.json` — column metadata

Training does **not** retrain FinBERT — Stage 1 is frozen; only Stage 2 weights are learned.

### 5.4 Stage 1B — News-adapted FinBERT (`src/train_finbert_news.py`)

Multi-task fine-tune from PhraseBank checkpoint on filtered news (train ≤ 2019):

- **Direction loss:** excess Up/Down (weight 1.0)
- **Sentiment distillation:** KL to PhraseBank teacher probs (weight 0.5)

Exports 3-class sentiment checkpoint to `models/finbert_news_adapted/`.

### 5.5 Stage 2 v2 — LightGBM ablations (`src/train_price_model.py --version v2`)

**Temporal split:**

| Set | Dates | Purpose |
|-----|-------|---------|
| Train | ≤ 2019-09-30 | Fit model |
| Validation | 2019-10-01 – 2019-12-31 | MCC threshold tuning |
| Test | ≥ 2020-01-01 | Held-out eval |

**Ablations (per FinBERT variant):** `sentiment_only`, `market_only`, `sentiment_market`, `full_fusion`

**Estimator:** `LightGBM` + `CalibratedClassifierCV(isotonic)`; PCA(**64**) on CLS for fusion configs.

**Threshold:** MCC-optimal on validation (saved in `pipeline.pkl`).

**Artifacts:** `models/price_direction_v2/{variant}_{ablation}/` + `best_model.json`

**Reference result:** Best val MCC **0.137** (`news_full_fusion`); 2020 test MCC remains weak (~−0.06) due to regime shift — see `metrics_module3_v2.json` subset metrics.

---

## 6. Phase 3 — Sentiment Evaluation (`src/evaluate.py --module sentiment`)

Evaluate all three models (Naive Bayes, SVM, FinBERT) on the held-out PhraseBank test set.

**Metrics:** accuracy, macro/weighted precision/recall/F1, per-class metrics, confusion matrices.

**Primary model selection metric:** **macro F1**

**Outputs:**

- `evaluation/metrics_module1.json` — Naive Bayes and SVM
- `evaluation/metrics_module2.json` — FinBERT
- `evaluation/figures/confusion_matrix_{model}.png`

FinBERT is the Stage 1 production model for both the demo and the price pipeline.

---

## 7. Phase 4 — Price Direction Evaluation

### v1 (`python src/evaluate.py --module market`)

See `evaluation/metrics_module3.json` — raw next-day return direction.

### v2 (`python src/evaluate.py --module market --version v2`)

**Target:** excess return vs SPY on **company-filtered** headlines (2020 test, ~1.4k rows).

**Reports:** optimal vs default threshold, all ablation MCCs, baselines (`always_up`, `momentum_excess`, `sentiment_only`), subset metrics (relevant-only, large moves, non-neutral).

**Outputs:**

- `evaluation/metrics_module3_v2.json`
- `evaluation/figures/confusion_matrix_price_model_v2.png`
- `evaluation/figures/calibration_curve_v2.png`
- `evaluation/figures/ablation_comparison.png`

### Limitations

- Predicts **next-day excess return vs SPY**, not intraday or causal impact
- 2020 test period (COVID) differs from 2019 validation — expect val/test gap
- Company filter reduces macro headline noise but also reduces sample size

---

## 8. Phase 5 — Streamlit Demo (`app/app.py`)

**Framework:** Streamlit

**Models loaded:** FinBERT (Stage 1) + v2 best LightGBM pipeline from `models/price_direction_v2/best_model.json`

**UI workflow:**

1. Text area for headline input + **Analyze** button
2. **Ticker select** (top 19) and **trading date** (2018–2020 range)
3. **Stage 1:** sentiment label + probability bars
4. **Stage 2:** predicted **excess return direction vs SPY** (`Up`/`Down`), `P(Up)`, `P(Down)`, confidence
5. Low-confidence warning if confidence < 0.55
6. Sidebar: excess return ground truth from v2 phrasebank dataset when available

**Run:**

```bash
streamlit run app/app.py
```

### Demo test cases (v2 pipeline)

Use these with the Streamlit demo after v2 retrain. Set **ticker** and **date** in the sidebar, paste the headline, click **Analyze**. Sidebar ground truth comes from `price_model_dataset_v2_phrasebank.parquet` (excess return vs SPY).

**v2 notes:**
- Stage 2 predicts **excess return vs SPY** (beat market / underperform), not raw price direction.
- Stage 1 inference uses **news-adapted FinBERT** when best model is `news_full_fusion`.
- Decision threshold is **~0.425** (MCC-tuned on 2019 Q4) — many outputs are **Down** when P(Up) < 42.5%.
- Headlines should mention the **company or ticker** to match training (company filter).

#### Part A — Stage 1 sanity checks (news-style, company-relevant)

| # | Ticker | Date | Headline | Expected Stage 1 |
|---|--------|------|----------|------------------|
| A1 | NVDA | 2020-03-16 | Nvidia shares surge after company reports record datacenter revenue growth | positive (pos ≈ 100%) |
| A2 | MRK | 2019-05-10 | Merck wins FDA approval for new oncology drug candidate | positive (pos ≈ 100%) |
| A3 | HD | 2020-04-21 | Home Depot beats earnings estimates and raises outlook for the year | positive (pos ≈ 100%) |
| A4 | BABA | 2020-06-10 | Alibaba shares plunge on delisting fears and regulatory crackdown concerns | negative (neg ≈ 59%) |
| A5 | NFLX | 2020-01-15 | Netflix subscriber growth misses analyst expectations in latest quarter | negative (neg ≈ 64%) |
| A6 | WFC | 2020-02-05 | Wells Fargo faces heavy fines over sales practice violations | negative (neg ≈ 53%) |
| A7 | JNJ | 2019-08-20 | Johnson and Johnson reaffirms full year guidance unchanged from prior quarter | neutral (neu ≈ 70%) |

#### Part B — Stage 2 behavior (excess return; sentiment ≠ direction)

Most v2 outputs land near **50–53% confidence** with **Down** predicted. Positive sentiment does **not** imply excess Up.

| # | Ticker | Date | Headline | Expected Stage 2 |
|---|--------|------|----------|------------------|
| B1 | NVDA | 2020-03-16 | (A1 headline) | Down, P(Up) ≈ 50%, conf ≈ 50%, warning |
| B2 | BABA | 2020-06-10 | (A4 headline) | Down, P(Up) ≈ 47%, conf ≈ 53%, warning |
| B3 | MRK | 2019-05-10 | (A2 headline) | Down, P(Up) ≈ 47%, conf ≈ 53%, warning |
| B4 | NFLX | 2019-07-18 | Large Option Trades Hint At How Institutions Are Playing The Netflix Dip | Down, P(Up) ≈ 26%, **conf ≈ 74%** |

B4 shows the model can be confident — but about **underperforming SPY**, even on a positive headline.

#### Part C — Real cached headlines (sidebar excess ground truth)

Paste **exact** headlines; sidebar shows `excess_forward_return` and excess direction.

| # | Ticker | Date | Headline (paste exactly) | Expected Stage 1 | Actual excess (sidebar) | Stage 2 (approx.) |
|---|--------|------|---------------------------|------------------|-------------------------|-------------------|
| C1 | BABA | 2020-06-09 | Alibaba To Add 5,000 Workers To Its Cloud Division As It Plans $28B Investment | positive | **Up** (+1.9%) | Down ~50% |
| C2 | NVDA | 2020-05-28 | Synopsys' Silicon-Proven DesignWare DDR IP for High-Performance Cloud Computing Networking Chips Selected by NVIDIA | neutral | **Up** (+4.1%) | Down ~50% |
| C3 | MRK | 2020-06-10 | The Daily Biotech Pulse: Keytruda Setback For Merck, Denali Pulls The Plug On Neurological Asset | positive | **Up** (+0.4%) | Down ~53% |
| C4 | NFLX | 2019-07-18 | Large Option Trades Hint At How Institutions Are Playing The Netflix Dip | positive | **Down** (−2.6%) | Down ~74% conf |

C1–C3: model predicts Down while actual excess was Up (2020 generalization gap). C4: confident Down and **correct**.

#### Part D — Gotcha cases

| # | Ticker | Date | Headline | What happens | Lesson |
|---|--------|------|----------|--------------|--------|
| D1 | BABA | 2020-06-10 | How Delisting Chinese Stocks Could Hurt Wall Street | **Not in v2 dataset** (no BABA/Alibaba in text) — sidebar likely empty | v2 trains on company-relevant headlines only |
| D2 | BABA | 2020-06-10 | BABA stock price is determined to rise in the upcoming days | S1 positive (~56% pos), S2 Down ~52% | Forecast wording; weak excess signal |
| D3 | BABA | 2020-06-10 | Alibaba shares surge on supportive Chinese tech policies | S1 positive (~100%), S2 Down ~53% | Bullish tone ≠ beat SPY tomorrow |
| D4 | BABA | 2020-06-10 | Chinese government announces supportive policies for tech sector | S1 positive, **no company name** | Macro headline — poor fit for single-ticker v2 |

#### Suggested 5-minute v2 demo flow

1. **A1** — strong positive sentiment
2. **B1** — same headline, Stage 2 Down at ~50% (sentiment ≠ excess)
3. **C4** — cached headline where confident Down matches sidebar
4. **D1 vs D3** — macro vs company-relevant contrast
5. **B4** — high-confidence Down on positive Netflix headline

---

## 9. Configuration (`config.yaml`)

```yaml
data:
  phrasebank_path: "data/raw/PhraseBank/data.csv"
  news_path: "data/raw/DailyFinancialNews/analyst_ratings_processed.csv"
  aligned_news_prices_path: "data/processed/aligned_news_prices.csv"
  prices_daily_path: "data/external/prices_daily.parquet"
  spy_prices_path: "data/external/spy_prices.parquet"
  price_model_dataset_path: "data/processed/price_model_dataset.parquet"
  news_start_date: "2018-01-01"
  news_tickers: [MRK, MS, MU, NVDA, QQQ, M, EBAY, NFLX, GILD, VZ, DAL, JNJ, QCOM, BABA, KO, ORCL, FDX, HD, WFC]
  price_start_date: "2018-01-01"
  price_end_date: "2020-06-11"
  spy_end_date: "2020-06-12"
  forward_flat_threshold: 0.001
  price_train_end_date: "2019-12-31"
  price_test_start_date: "2020-01-01"
  random_seed: 42

models:
  finbert:
    pretrained_model: "yiyanghkust/finbert-tone"
    max_length: 128
    batch_size: 16
    save_path: "models/finbert_finetuned/"
  price_direction:
    save_path: "models/price_direction/"
    pca_components: 32
    logistic_c: 1.0
    calibration_method: "isotonic"
    calibration_cv: 3
    confidence_threshold: 0.55

evaluation:
  metrics_module3_path: "evaluation/metrics_module3.json"
  figures_path: "evaluation/figures/"
```

---

## 10. Requirements and Setup

### Requirements

- **Language:** Python 3.9+
- **Packages** (`requirements.txt`): `numpy`, `pandas`, `scikit-learn`, `torch`, `transformers`, `accelerate`, `yfinance`, `matplotlib`, `seaborn`, `streamlit`, `pyyaml`, `python-dotenv`, `tqdm`, `sentencepiece`, `joblib`, `pyarrow`

### Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Place raw data at:

- `data/raw/PhraseBank/data.csv`
- `data/raw/DailyFinancialNews/analyst_ratings_processed.csv`

Notebooks are for interactive EDA only. The production pipeline runs via `src/` scripts.

---

## 11. Execution Order

Run scripts from the project root in this order:

```bash
# Phase 1 — Data pipeline
python src/preprocess.py
python src/align_market.py
python src/build_price_dataset.py

# Phase 2 — Modeling
python src/train_baseline.py        # optional for Phase 3 comparison
python src/train_finbert.py         # Stage 1 (required)
python src/train_price_model.py     # Stage 2

# Phase 3 — Sentiment evaluation
python src/evaluate.py --module sentiment

# Phase 4 — Price direction evaluation
python src/evaluate.py --module market          # v1
python src/evaluate.py --module market --version v2

# Phase 5 — Demo
streamlit run app/app.py
```

### v2 full retrain (recommended)

```bash
python src/preprocess.py
python src/align_market.py
python src/train_finbert.py                     # Stage 1A (skip if checkpoint exists)
python src/train_finbert_news.py              # Stage 1B (~30 min CPU)
python src/build_price_dataset.py --version v2 --finbert-variant phrasebank
python src/build_price_dataset.py --version v2 --finbert-variant news
python src/train_price_model.py --version v2
python src/evaluate.py --module market --version v2
streamlit run app/app.py
```

**Note:** `build_price_dataset.py` runs FinBERT inference and CLS embedding extraction on ~16k rows; allow several minutes on CPU.
