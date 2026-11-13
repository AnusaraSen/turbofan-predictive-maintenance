# Test Suite

Automated smoke and unit tests for the C-MAPSS turbofan RUL prediction pipeline and FastAPI service, located in [`test_pipeline_smoke.py`](test_pipeline_smoke.py).

## How to Run

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Or with `uv`:

```powershell
uv run pytest -v
```

## What Is Tested

### 1. Scoring (`src/scoring.py`)
- **`test_rmse_and_phm08_worked_examples`**: Verifies `rmse` and `phm08_score` against exact worked values (`d = -30, -10, +10, +30`) and confirms asymmetric penalties (`+d` penalized more heavily than `-d`).
- **`test_evaluate_and_per_engine_breakdown`**: Verifies `evaluate` output keys (`RMSE`, `PHM08_score`) and checks that `phm08_per_engine` contributions sum to the total `PHM08_score`.

### 2. Preprocessing (`src/preprocessing.py`)
- **`test_compute_rul_piecewise_cap`**: Verifies piecewise-linear RUL countdown to `0` at failure and capping at `rul_cap=125`.
- **`test_regime_clustering_normalization_and_sensor_selection`**:
  - Fits `KMeans` (`n_regimes=6`) and checks `regime_id` assignment.
  - Verifies `select_informative_sensors` drops sensors that shift across regimes but are constant *within* each regime (and drops globally constant sensors).
  - Verifies `fit_regime_norm_stats` + `normalize_by_regime` produce zero-mean per-regime values with no `NaN`s.

### 3. Sequence Construction (`src/sequence_builder.py`)
- **`test_train_and_test_sequence_shapes_and_padding`**:
  - Confirms `build_train_sequences` skips engines shorter than `window_length` and aligns labels with the final cycle of each window.
  - Confirms `build_test_sequences` extracts the last window for long engines and applies front zero-padding for short engines.
- **`test_simulate_test_sequences_short_engine_edge_case`**: Verifies synthetic short-engine and early-cutoff edge cases produce valid `(n_units, window_length, n_sensors)` arrays without negative-index wraparound or `NaN`s.

### 4. Feature Engineering (`src/feature_engineering.py`)
- **`test_feature_engineering_no_nans_and_no_cross_engine_leakage`**: Confirms `build_baseline_features` generates rolling, lag, and rate-of-change columns with no `NaN`s and no cross-engine leakage between consecutive `unit_number` groups.

### 5. FastAPI Service (`api/main.py`)
- **`test_api_predict_and_feature_importance`**: Uses FastAPI's `TestClient` against the live app and saved `FD001` artifacts:
  - `POST /predict` returns `200` with `predicted_RUL` and `capped` for both `gbr` and `lstm`.
  - `POST /predict` returns `400` for invalid `dataset_name` or `model_type`.
  - `GET /feature-importance/FD001` returns `200` with ranked features (`400` for an invalid dataset).
