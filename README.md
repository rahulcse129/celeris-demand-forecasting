# Celeris Automobili Industrial — Demand Forecasting & Planning Control Tower
## MKS AI & Data Hackathon 2026 Solution

An end-to-end industrial machine learning forecasting system and planning control tower for **Celeris Automobili Industrial**, covering 600 SKUs across 7 product families and 6 global regions for the 19-month forecast horizon (**November 2026 – May 2028**).

---

## 🏆 Architecture & Solution Highlights

```
Raw ERP & Reference Data
        │
        ▼
[ Phase 1: Data Cleaning & Hygiene ] (src/clean.py)
   ├── 29 region variants mapped to 6 canonical regions
   ├── 220 noisy customer names normalized to 12 canonical OEM accounts via Levenshtein matching
   ├── Date format harmonization & anomaly repair (e.g. 2062 -> 2026 typo fixes)
   ├── Duplicate ERP transactions resolved by latest Load_Timestamp
   ├── Unit scaling & revenue mismatches reconciled
   └── Unconstrained demand reconstructed using inventory stockout requests
        │
        ▼
[ Phase 2: Anomaly Detection & Business Signals ] (src/anomalies.py)
   ├── Statistical Outliers (IQR filtering per SKU-Region time series)
   ├── Structural Breaks (Macro shocks, COVID, semiconductor supply bottlenecks)
   ├── Pre-Buy Surge patterns prior to annual January price adjustments
   ├── Supply constraints & unfulfilled order volume tracking
   └── Customer account switches & lifecycle boundary alerts
        │
        ▼
[ Phase 3: Advanced Feature Engineering ] (src/features.py)
   ├── Calendar & Cyclical signals (sine/cosine month encoding, Q4/December flags)
   ├── Historical hierarchical summaries (SKU-Region, SKU overall, Product Family baselines)
   ├── Dynamic multi-period lags (1, 2, 3, 6, 12 months) & rolling statistics (mean, std, min, max)
   ├── Croston-SBA Intermittent Demand Rate calculation (Syntetos-Boylan Approximation)
   ├── Syntetos-Boylan intermittency categorization (Smooth, Erratic, Intermittent, Lumpy)
   ├── Macroeconomic indicator lag structures (PMI, Auto production YoY, EV adoption, Raw materials)
   ├── Business event flags (Price changes, Disruptions, Holidays, Supply constraints)
   ├── Powertrain transition dynamics (EV Adoption interaction with ICE vs EV powertrain SKUs)
   └── Expert NPI: Inherited demand & seasonal lags for new launches via Replaces_SKU
        │
        ▼
[ Phase 4: Multi-Model Ensemble & Hierarchical Reconciliation ] (src/models.py)
   ├── Baseline 1: Seasonal Naïve (Lag-12) — WAPE: 0.4163, Bias: +0.0190
   ├── Baseline 2: 6-Month Moving Average — WAPE: 0.5369, Bias: -0.0311
   ├── Model A: LightGBM Global Regressor — WAPE: 0.3157, Bias: +0.0102
   ├── Model B: CatBoost Regressor — WAPE: 0.3000, Bias: -0.0432
   ├── Final Blended Ensemble (55% LightGBM + 45% CatBoost) — WAPE: 0.3058, Bias: -0.0130 (Near Zero!)
   ├── Probabilistic Bounds: Quantile Regression (P10, P50, P90)
   ├── Hierarchical Reconciliation: Top-down / bottom-up coherence verification across Product Family & Region
   └── Lifecycle Post-Processing: EOL phaseout cutoffs (156 rows) & NPI successor ramp-over
        │
        ▼
[ Phase 5: Planning Control Tower Dashboard ] (dashboard/app.py)
   ├── 1. Executive Overview (Continuous history-to-forecast timeline, mix by family, region, channel)
   ├── 2. Data Health (Pre vs Post cleaning auditing, issues breakdown)
   ├── 3. Demand Explorer (Drill-downs, seasonality heatmap, intermittent demand profiling)
   ├── 4. Anomaly Monitor (Categorized anomaly register, event attribution)
   ├── 5. Forecast & Accuracy (Multi-model benchmark, backtest WAPE/Bias, hierarchical coherence)
   ├── 6. Macro Heatmap (Multi-lag correlation matrix across product families)
   ├── 7. Lifecycle View (Active catalog, EOL countdown, NPI transition tracking via Replaces_SKU)
   ├── 8. What-If Scenario Simulator (Real-time macro stress testing on PMI, Auto production, EV index)
   ├── 9. Procurement Plan (P90) (Safety stock buffers & procurement volume recommendations)
   └── 10. Planner Exceptions & Lost Sales (High-volatility alerts, stockout root cause, CSV export center)
```

---

## 📊 Benchmark Validation Results

| Model | Model Class | Backtest WAPE (Lower is Better) | Forecast Bias (Near 0 is Ideal) | MAE | Relative Improvement vs Baseline |
|---|---|---|---|---|---|
| **Seasonal Naïve (Lag-12)** | Benchmark Baseline | `0.4163` | `+0.0190` | `19.41` | — |
| **6-Month Moving Average** | Benchmark Baseline | `0.5369` | `-0.0311` | `25.04` | -28.9% |
| **LightGBM Global Regressor** | Gradient Boosted Trees | `0.3157` | `+0.0102` | `14.70` | **+24.2%** |
| **CatBoost Regressor** | Categorical Boosting | `0.3000` | `-0.0432` | `13.98` | **+27.9%** |
| **Blended Ensemble (Final)** | Dual Gradient Boosters | **`0.3058`** | **`-0.0130`** | **`14.24`** | **+26.5%** |

*Validation window: Rolling 12-month backtest origin (`2025-11` to `2026-10`).*

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
python -m pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution

Run all workstreams sequentially:

```bash
# 1. Clean raw ERP extracts and build data quality audit
python src/clean.py

# 2. Extract anomaly register and classify supply/demand events
python src/anomalies.py

# 3. Engineer lag, rolling, macro, and lifecycle features
python src/features.py

# 4. Train ensemble models, evaluate backtest, and generate submission.csv
python src/models.py
```

### 3. Launch the Interactive Control Tower Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 File Structure

```
f:\Project\IMS-KS\
├── dataset\student_data\          # Raw competition data
│   ├── train.csv
│   ├── test.csv
│   ├── sku_master.csv
│   ├── macro_indicators.csv
│   ├── events_calendar.csv
│   ├── inventory_stockouts.csv
│   └── sample_submission.csv
├── data\
│   ├── cleaned\                   # Cleaned and reconciled datasets
│   ├── features\                  # Feature-engineered matrices & intermittency classes
│   └── forecasts\                 # Forecast details, P10/P50/P90, backtest results
├── src\
│   ├── utils.py                   # Canonical entity mappings & Levenshtein matching
│   ├── clean.py                   # Production data cleaning pipeline
│   ├── anomalies.py               # Anomaly detection & register generator
│   ├── features.py                # Feature engineering pipeline (Croston-SBA, NPI inheritance)
│   ├── models.py                  # LightGBM + CatBoost ensemble, reconciliation & quantiles
│   └── generate_notebook.py       # Automated Jupyter walkthrough notebook generator
├── dashboard\
│   └── app.py                     # Streamlit Planning Control Tower dashboard (10 views)
├── notebooks\
│   └── celeris_demand_forecasting_walkthrough.ipynb  # Comprehensive jury demonstration
├── reports\
│   ├── data_quality_report.json   # Machine-readable cleaning audit
│   ├── anomaly_register.csv       # Anomaly event register
│   └── model_metrics.json         # WAPE, Bias, MAE performance summary
├── submission.csv                 # Verified competition submission file (58,219 rows)
├── requirements.txt
└── README.md
```

---

## 📈 Submission Integrity & Compliance

- **Submission File:** `submission.csv`
- **Rows:** 58,219 rows (exactly matching `test.csv`)
- **Columns:** `Date`, `SKU_ID`, `Region`, `Units_Sold_Forecast`
- **Constraints:** Zero nulls, non-negative integer demand constraints satisfied.
- **Total 19-Month Forecast Volume:** **2,657,218 units** (Mean: ~45.6 units/SKU-month)
