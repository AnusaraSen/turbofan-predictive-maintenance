# Predictive Maintenance for Turbofan Engines — C-MAPSS RUL Prediction

Predicts Remaining Useful Life (RUL) for aircraft turbofan engines using NASA's C-MAPSS dataset, comparing a scikit-learn Gradient Boosting Regressor baseline against a TensorFlow/Keras stacked LSTM sequence model — evaluated across all four C-MAPSS sub-datasets (FD001–FD004) using both RMSE and the domain-specific, asymmetric PHM08 scoring function.

## Why this project

Most public C-MAPSS implementations pick one sub-dataset, use a single model (usually just an LSTM), and evaluate with plain RMSE. This project deliberately does more:

- **Two model families compared head-to-head** — a feature-engineered scikit-learn baseline vs. a raw-sequence TensorFlow/Keras LSTM, not just one architecture.
- **All four sub-datasets**, not just FD001 — including the harder multi-regime (FD002/FD004) and multi-fault-mode (FD003/FD004) variants.
- **The domain-appropriate PHM08 asymmetric scoring function**, not just RMSE — because in real maintenance, overestimating remaining life is far more dangerous than underestimating it.
- **Deployment** — a FastAPI prediction endpoint and Streamlit dashboard, not just a notebook.
- **Documented, evidence-based debugging and limitation analysis** — every non-trivial design decision below was validated against real data, and known weaknesses are reported honestly rather than hidden.

## Problem framing

Each C-MAPSS sub-dataset contains multiple turbofan engines, each run from a healthy state to failure (training set) or truncated at some point before failure (test set). Given sensor readings up to the truncation point, the goal is to predict **Remaining Useful Life (RUL)** — the number of operating cycles left before failure.

| Dataset | Train units | Test units | Operating conditions | Fault modes |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1 (sea level) | 1 (HPC degradation) |
| FD002 | 260 | 259 | 6 | 1 (HPC degradation) |
| FD003 | 100 | 100 | 1 (sea level) | 2 (HPC + Fan degradation) |
| FD004 | 248 | 249 | 6 | 2 (HPC + Fan degradation) |

## Methodology

### Preprocessing (parameterized across all four datasets, one codebase)

- **Piecewise RUL labeling**, capped at 125 cycles. Degradation isn't detectable from sensors early in an engine's life, so capping avoids teaching the model unlearnable noise from the flat, healthy portion of each trajectory. (See [Known Limitations](#known-limitations) for the tradeoff this introduces.)
- **Regime clustering + per-regime normalization.** FD002/FD004 operate under 6 distinct flight conditions; a sensor's raw value reflects both flight condition and degradation. K-means (k=6) clusters rows by operating condition, and each sensor is z-score normalized *within* its own regime — removing condition-driven variance so what remains is closer to a pure degradation signal. For FD001/FD003 (1 condition), this collapses to k=1, i.e. plain global normalization — one pipeline, no special-casing.
- **Sensor selection, computed post-clustering.** Near-constant sensors are dropped based on variance *within* each regime, averaged across regimes — not raw global variance. This matters specifically for FD002/FD004: a sensor like ambient inlet temperature can look highly variable across the whole dataset (because it shifts with flight condition) while being genuinely flat within any single regime. Selecting on raw variance alone would incorrectly keep these uninformative sensors.
- **Sliding-window sequence construction** (window length = 30 cycles) for the LSTM. Training slides across each engine's full history, generating many overlapping windows per engine (e.g., 17,731 training sequences from FD001's 100 engines). Test/inference uses one window per engine — the most recent 30 cycles — with zero-padding at the front for any engine shorter than the window.
- **Rolling/lag/rate-of-change feature engineering** for the GBR baseline, computed strictly within each engine's own history (grouped by unit, sorted by cycle) to avoid cross-engine leakage.

### Evaluation

- **RMSE** — standard regression error, in cycles.
- **PHM08 asymmetric scoring function** — penalizes late/optimistic RUL predictions (telling maintenance "you have more time than you do") far more heavily than early/conservative ones, reflecting real-world risk asymmetry:

  ```
  d = predicted_RUL - actual_RUL
  if d < 0:  score = exp(-d/13) - 1   (conservative — cheaper)
  if d ≥ 0:  score = exp(d/10) - 1    (optimistic — expensive)
  ```

- **Two evaluation stages**, kept clearly separate throughout development:
  1. **Internal validation** — 20% of training engines held out by unit (never mixed across train/val), with a simulated random truncation point per engine to mimic the real test set's structure.
  2. **Real test set** — the official `test_FDxxx.txt` + `RUL_FDxxx.txt` benchmark, evaluated once as the final, literature-comparable result. Predictions matched to true RUL by `unit_number`, never row order.

## Results

Evaluated on the official C-MAPSS test sets (`test_FD001`–`FD004` + `RUL_FD001`–`FD004`) using `src/final_eval.py` and recorded in [`results/metrics_summary.csv`](results/metrics_summary.csv):

### Overall test-set results

| Dataset | Model | RMSE | PHM08 score |
|---|---|---|---|
| FD001 | GBR | 19.48 | 1527.99 |
| FD001 | LSTM | **15.51** | **390.48** |
| FD002 | GBR | 30.21 | 12624.86 |
| FD002 | LSTM | **26.32** | **7119.76** |
| FD003 | GBR | 21.14 | 2096.33 |
| FD003 | LSTM | **16.00** | **1137.45** |
| FD004 | GBR | 31.78 | 9873.21 |
| FD004 | LSTM | **27.67** | **6426.67** |

### Within RUL-cap (`true RUL ≤ 125`) breakdown

Because both models are trained on RUL capped at 125 cycles, healthy test engines truncated early (`true RUL > 125`) dominate the blended `FD002`/`FD004` metrics (see [Known Limitations](#known-limitations)). Evaluating strictly within the observable degradation window (`RUL ≤ 125`):

| Dataset | Within-cap units | GBR RMSE | LSTM RMSE | GBR PHM08 | LSTM PHM08 |
|---|---|---|---|---|---|
| FD001 | 89 / 100 | 19.13 | **13.93** | 1476.70 | **324.67** |
| FD002 | 202 / 259 | 19.14 | **15.73** | 2209.84 | **1442.55** |
| FD003 | 85 / 100 | 21.28 | **15.07** | 2033.21 | **1074.85** |
| FD004 | 181 / 248 | 22.51 | **14.70** | 3936.57 | **845.02** |

## Known limitations

### RUL-cap ceiling effect (FD002 / FD004)

Both models are trained on RUL labels capped at 125 cycles. This creates a structural ceiling: for test engines truncated while still healthy (true RUL > 125), **neither model can predict above ~125**, regardless of architecture — this was confirmed by checking that both the GBR and LSTM independently plateau near the same value on these engines.

On FD004, 67 of 248 test engines (27%) have true RUL exceeding the cap, and these engines dominate the blended PHM08 score for both models — masking the LSTM's real advantage, which only becomes visible once results are split by this boundary (see Results above). This is a shared limitation of the labeling strategy, not a weakness specific to either model, and is a known, accepted tradeoff in the RUL-capping literature: precise remaining life is only meaningful — and only learnable from the sensor data — as failure approaches.

### Run-to-run LSTM variance

An early un-seeded LSTM training run on FD004 produced meaningfully different results across repeated runs (PHM08 swinging from ~1937 to ~1222) due to unseeded weight initialization and batch shuffling. This was diagnosed and fixed by explicitly seeding NumPy, Python's `random`, and TensorFlow at the start of every training run; results are now reproducible run-to-run. FD001/FD002/FD003 LSTM results predating this fix are being re-verified.

## Project structure

```
turbofan-predictive-maintenance/
├── data/
│   ├── raw/                     # C-MAPSS train/test/RUL .txt files (FD001–FD004)
│   └── processed/
├── src/
│   ├── config.py                # paths, sensor names, per-dataset metadata (n_regimes, etc.)
│   ├── data_loader.py           # reads raw txt files into DataFrames
│   ├── preprocessing.py         # regime clustering, normalization, RUL labeling, sensor selection
│   ├── feature_engineering.py   # rolling/lag/rate-of-change features for the GBR baseline
│   ├── sequence_builder.py      # sliding-window sequences for the LSTM
│   ├── scoring.py               # RMSE + PHM08, including per-engine diagnostic breakdown
│   ├── train_baseline.py        # GBR training, unit-based split, simulated truncation
│   ├── train_lstm.py            # LSTM training, same split/evaluation methodology
│   ├── final_eval.py            # evaluates GBR + LSTM across FD001–FD004 -> results/metrics_summary.csv
│   ├── save.py                  # exports sample_engines.csv and trained model artifacts
│   ├── sample_RUL_attach.py     # attaches true RUL to sample_engines.csv
│   └── generate_json.py         # generates sample JSON request payload for the API
├── models/                      # saved GBR/LSTM artifacts & sample_engines.csv (FD001–FD004)
├── results/
│   ├── metrics_summary.csv      # overall RMSE and PHM08 summary across FD001–FD004
│   └── plots/
├── notebooks/                   # EDA and result-inspection notebooks
├── api/
│   └── main.py                  # FastAPI /predict and /feature-importance endpoints (with RUL-cap flagging)
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── tests/
│   ├── test_pipeline_smoke.py   # unit & API smoke tests (incl. short-engine edge cases)
│   └── README.md                # test suite documentation
├── pyproject.toml
└── uv.lock
```

## Setup & Usage

This project uses Python 3.11+ and `uv` (`pyproject.toml` + `uv.lock`):

```powershell
uv sync
.\.venv\Scripts\Activate.ps1
```

If you need to download the C-MAPSS dataset, fetch it from the [NASA Prognostics Center of Excellence data repository](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip) and place the 12 extracted `.txt` files (`train_/test_/RUL_` × `FD001`–`FD004`) under `data/raw/`.

```powershell
# Sanity-check data loading
cd src
python data_loader.py

# Evaluate saved models across FD001–FD004 and update results/metrics_summary.csv
python final_eval.py

# Run unit & API smoke tests (from project root)
cd ..
pytest -v
```

### Run the API & Dashboard

**Terminal 1 — FastAPI backend** (from project root):
```powershell
uvicorn api.main:app --reload
```

**Terminal 2 — Streamlit dashboard** (from `dashboard/`):
```powershell
cd dashboard
streamlit run app.py
```

## Tech stack

**Modeling:** Python, pandas, NumPy, scikit-learn (Gradient Boosting Regressor), TensorFlow/Keras (stacked LSTM)
**Evaluation:** custom RMSE + PHM08 asymmetric scoring implementation, verified against reference worked examples
**Deployment:** FastAPI (prediction endpoint with RUL-cap flagging), Streamlit (dashboard)

## Engineering notes worth highlighting

A handful of real bugs were caught and fixed during development through deliberate validation at every step, rather than trusting output at face value:

- A numpy negative-index wraparound in the test-truncation sequence builder, which would have silently produced empty/invalid windows for engines with an early random cutoff but long total life — caught via a synthetic edge-case test before it ever touched real evaluation numbers.
- A sensor-selection bug where checking variance *after* z-score normalization made every non-exactly-constant sensor look equally "informative" (since z-scoring forces unit variance on anything with nonzero variance) — fixed by selecting on raw, pre-normalization variance instead.
- A within-regime vs. global variance issue specific to FD002/FD004: sensors that are physically constant within any single operating regime can show high *raw* variance simply because the regime itself changes — fixed by computing variance within each regime and averaging, rather than pooling all regimes together.
- Hardcoded `n_regimes=1` left over from initial FD001-only development, silently ignored FD002/FD004's actual 6-regime structure until caught by an explicit `regime_id.nunique()` check.

## Future work

- Multi-run statistical reporting (mean ± std) for LSTM results, given observed run-to-run variance
- Ablation on RUL cap value itself (e.g., 125 vs. 150) as a documented experiment rather than a fixed assumption
- Containerization (`Dockerfile` / `docker-compose.yml`) for one-command API + dashboard deployment