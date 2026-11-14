import os
import joblib
from pathlib import Path
import keras
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import PROJECT_ROOT
from data_loader import load_rul_data
from feature_engineering import build_baseline_features
from preprocessing import assign_regimes, normalize_by_regime
from scoring import evaluate
from sequence_builder import build_test_sequences
from train_baseline import get_last_cycle_per_unit


app = FastAPI(title="Turbofan RUL Prediction API")

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
artifacts = {}
fleet_cache: dict[str, list[dict]] = {}
trajectory_cache: dict[tuple[str, int], list[dict]] = {}
breakdown_cache: list[dict] | None = None

for dataset_name in DATASETS:
    dataset_dir = PROJECT_ROOT / "models" / dataset_name
    artifacts[dataset_name] = {
        "gbr_model": joblib.load(dataset_dir / "gbr_model.pkl"),
        "kmeans_gbr": joblib.load(dataset_dir / "kmeans.pkl"),
        "informative_sensors_gbr": joblib.load(dataset_dir / "informative_sensors.pkl"),
        "feature_cols": joblib.load(dataset_dir / "feature_cols.pkl"),
        "lstm_model": keras.models.load_model(dataset_dir / "lstm_model.keras"),
        "kmeans_lstm": joblib.load(dataset_dir / "kmeans_lstm.pkl"),
        "informative_sensors_lstm": joblib.load(dataset_dir / "informative_sensors_lstm.pkl"),
        "norm_stats": joblib.load(dataset_dir / "norm_stats.pkl"),
    }

print(f"Loaded artifacts for: {list(artifacts.keys())}")

class SensorReading(BaseModel):
    time_cycles: int
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float

class PredictRequest(BaseModel):
    dataset_name: str       # e.g. "FD001" - explicit, not guessed
    model_type: str         # "gbr" or "lstm"
    unit_number: int
    readings: List[SensorReading]

class PredictResponse(BaseModel):
    unit_number: int
    dataset_name: str
    model_type: str
    predicted_RUL: float
    capped: bool
    regime_warning: Optional[str] = None
    note: Optional[str] = None

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if request.dataset_name not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {request.dataset_name}")

    ds_artifacts = artifacts[request.dataset_name]

    df = pd.DataFrame([r.dict() for r in request.readings])
    df["unit_number"] = request.unit_number

    if request.model_type == "gbr":
        gbr_model = ds_artifacts["gbr_model"]
        kmeans_gbr = ds_artifacts["kmeans_gbr"]
        informative_sensors_gbr = ds_artifacts["informative_sensors_gbr"]
        feature_cols = ds_artifacts["feature_cols"]

        df_processed = assign_regimes(df, kmeans_gbr)
        df_processed = build_baseline_features(df_processed, informative_sensors_gbr)
        last_row = get_last_cycle_per_unit(df_processed)

        X = last_row[feature_cols]
        predicted_rul = float(gbr_model.predict(X)[0])

    elif request.model_type == "lstm":
        lstm_model = ds_artifacts["lstm_model"]
        kmeans_lstm = ds_artifacts["kmeans_lstm"]
        informative_sensors_lstm = ds_artifacts["informative_sensors_lstm"]
        norm_stats = ds_artifacts["norm_stats"]

        df_processed = assign_regimes(df, kmeans_lstm)
        all_sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
        df_processed = normalize_by_regime(df_processed, norm_stats, all_sensor_cols)

        X_seq, unit_ids = build_test_sequences(df_processed, informative_sensors_lstm, window_length=30)
        predicted_rul = float(lstm_model.predict(X_seq).flatten()[0])

    else:
        raise HTTPException(status_code=400, detail=f"Unknown model_type: {request.model_type}")

    capped = predicted_rul >= 124
    note = None
    if capped:
        note = "Prediction is at or near the model's training cap (125 cycles); engine is likely healthy, precise estimate beyond this point is not reliable."

    return PredictResponse(
        unit_number=request.unit_number,
        dataset_name=request.dataset_name,
        model_type=request.model_type,
        predicted_RUL=predicted_rul,
        capped=capped,
        note=note,
    )

@app.get("/feature-importance/{dataset_name}")
def feature_importance(dataset_name: str):
    if dataset_name not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset_name}")

    ds_artifacts = artifacts[dataset_name]
    gbr_model = ds_artifacts["gbr_model"]
    feature_cols = ds_artifacts["feature_cols"]

    importances = gbr_model.feature_importances_
    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    return {
        "dataset_name": dataset_name,
        "features": [{"feature": name, "importance": float(score)} for name, score in ranked],
    }


def _rul_status(rul: float) -> str:
    if rul < 25:
        return "Urgent"
    if rul < 75:
        return "Monitor"
    return "Healthy"


def _load_sample_df(dataset_name: str) -> pd.DataFrame:
    sample_path = PROJECT_ROOT / "models" / dataset_name / "sample_engines.csv"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"sample_engines.csv not found for {dataset_name}")
    return pd.read_csv(sample_path)


def _get_rul_map(dataset_name: str) -> dict[int, float]:
    try:
        rul_df = load_rul_data(dataset_name)
    except FileNotFoundError:
        return {}
    return {
        int(row["unit_number"]): float(row["RUL"])
        for _, row in rul_df.iterrows()
    }


@app.get("/health")
def health():
    return {"status": "ok", "datasets": DATASETS}


@app.get("/metrics")
def get_metrics():
    metrics_path = PROJECT_ROOT / "results" / "metrics_summary.csv"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="metrics_summary.csv not found")
    df = pd.read_csv(metrics_path)
    return {"metrics": df.to_dict(orient="records")}


@app.get("/datasets/{dataset_name}/engines")
def list_dataset_engines(dataset_name: str):
    if dataset_name not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset_name}")

    if dataset_name in fleet_cache:
        return {"dataset_name": dataset_name, "engines": fleet_cache[dataset_name]}

    sample_df = _load_sample_df(dataset_name)
    ds_artifacts = artifacts[dataset_name]

    # Batched GBR predictions
    df_gbr = assign_regimes(sample_df, ds_artifacts["kmeans_gbr"])
    df_gbr = build_baseline_features(df_gbr, ds_artifacts["informative_sensors_gbr"])
    last_rows = get_last_cycle_per_unit(df_gbr).sort_values("unit_number")
    gbr_preds = ds_artifacts["gbr_model"].predict(last_rows[ds_artifacts["feature_cols"]])
    gbr_by_unit = {
        int(uid): float(pred) for uid, pred in zip(last_rows["unit_number"], gbr_preds)
    }

    # Batched LSTM predictions
    df_lstm = assign_regimes(sample_df, ds_artifacts["kmeans_lstm"])
    all_sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    df_lstm = normalize_by_regime(df_lstm, ds_artifacts["norm_stats"], all_sensor_cols)
    X_seq, unit_ids = build_test_sequences(
        df_lstm, ds_artifacts["informative_sensors_lstm"], window_length=30
    )
    lstm_preds = ds_artifacts["lstm_model"].predict(X_seq, verbose=0).flatten()
    lstm_by_unit = {int(uid): float(pred) for uid, pred in zip(unit_ids, lstm_preds)}

    has_rul = "RUL" in last_rows.columns
    rul_map = _get_rul_map(dataset_name)
    engines = []
    for _, row in last_rows.iterrows():
        uid = int(row["unit_number"])
        gbr_rul = round(gbr_by_unit[uid], 2)
        lstm_rul = round(lstm_by_unit[uid], 2)
        if uid in rul_map:
            true_rul = round(rul_map[uid], 2)
        elif has_rul and pd.notna(row["RUL"]):
            true_rul = round(float(row["RUL"]), 2)
        else:
            true_rul = None
        engines.append({
            "unit_number": uid,
            "total_cycles": int(row["time_cycles"]),
            "gbr_rul": gbr_rul,
            "lstm_rul": lstm_rul,
            "true_rul": true_rul,
            "status": _rul_status(lstm_rul),
            "capped": bool(lstm_rul >= 124 or gbr_rul >= 124),
        })

    engines.sort(key=lambda e: e["lstm_rul"])
    fleet_cache[dataset_name] = engines
    return {"dataset_name": dataset_name, "engines": engines}


@app.get("/datasets/{dataset_name}/engines/{unit_id}")
def get_engine_detail(dataset_name: str, unit_id: int):
    if dataset_name not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset_name}")

    sample_df = _load_sample_df(dataset_name)
    engine_df = (
        sample_df[sample_df["unit_number"] == unit_id]
        .sort_values("time_cycles")
        .reset_index(drop=True)
    )
    if engine_df.empty:
        raise HTTPException(status_code=404, detail=f"Unit {unit_id} not found in {dataset_name}")

    ds_artifacts = artifacts[dataset_name]
    all_sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

    raw_with_regime = assign_regimes(engine_df, ds_artifacts["kmeans_lstm"])
    norm_df = normalize_by_regime(raw_with_regime, ds_artifacts["norm_stats"], all_sensor_cols)

    return {
        "dataset_name": dataset_name,
        "unit_number": unit_id,
        "total_cycles": int(len(engine_df)),
        "informative_sensors": ds_artifacts["informative_sensors_lstm"],
        "raw_rows": raw_with_regime.where(pd.notna(raw_with_regime), None).to_dict(orient="records"),
        "normalized_rows": norm_df.where(pd.notna(norm_df), None).to_dict(orient="records"),
    }


class TrajectoryRequest(BaseModel):
    dataset_name: str
    unit_number: int


@app.post("/predict/trajectory")
def predict_trajectory(request: TrajectoryRequest):
    if request.dataset_name not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {request.dataset_name}")

    cache_key = (request.dataset_name, request.unit_number)
    if cache_key in trajectory_cache:
        return {
            "dataset_name": request.dataset_name,
            "unit_number": request.unit_number,
            "points": trajectory_cache[cache_key],
        }

    sample_df = _load_sample_df(request.dataset_name)
    engine_df = (
        sample_df[sample_df["unit_number"] == request.unit_number]
        .sort_values("time_cycles")
        .reset_index(drop=True)
    )
    if engine_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Unit {request.unit_number} not found in {request.dataset_name}",
        )

    ds_artifacts = artifacts[request.dataset_name]

    # Batched GBR prediction at every cycle
    df_gbr = assign_regimes(engine_df, ds_artifacts["kmeans_gbr"])
    df_gbr = build_baseline_features(df_gbr, ds_artifacts["informative_sensors_gbr"])
    gbr_preds = ds_artifacts["gbr_model"].predict(df_gbr[ds_artifacts["feature_cols"]])

    # Batched LSTM prediction at every cycle (30-cycle sliding window with front zero-padding)
    df_lstm = assign_regimes(engine_df, ds_artifacts["kmeans_lstm"])
    all_sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    df_lstm = normalize_by_regime(df_lstm, ds_artifacts["norm_stats"], all_sensor_cols)
    inf_sensors = ds_artifacts["informative_sensors_lstm"]
    sensor_vals = df_lstm[inf_sensors].values

    window_length = 30
    windows = []
    for idx in range(len(df_lstm)):
        start = max(0, idx - window_length + 1)
        window = sensor_vals[start : idx + 1]
        if len(window) < window_length:
            pad = np.zeros((window_length - len(window), sensor_vals.shape[1]))
            window = np.vstack([pad, window])
        windows.append(window)

    X_seq = np.array(windows)
    lstm_preds = ds_artifacts["lstm_model"].predict(X_seq, verbose=0).flatten()

    has_rul = "RUL" in df_lstm.columns
    rul_map = _get_rul_map(request.dataset_name)
    end_rul = rul_map.get(request.unit_number)
    max_cycle = int(df_lstm["time_cycles"].max())

    points = []
    for idx in range(len(df_lstm)):
        cycle = int(df_lstm.loc[idx, "time_cycles"])
        gbr_rul = round(float(gbr_preds[idx]), 2)
        lstm_rul = round(float(lstm_preds[idx]), 2)
        if end_rul is not None:
            true_rul = round(end_rul + (max_cycle - cycle), 2)
        else:
            true_val = df_lstm.loc[idx, "RUL"] if has_rul else None
            true_rul = round(float(true_val), 2) if true_val is not None and pd.notna(true_val) else None
        points.append({
            "time_cycles": cycle,
            "regime_id": int(df_lstm.loc[idx, "regime_id"]),
            "gbr_rul": gbr_rul,
            "lstm_rul": lstm_rul,
            "true_rul": true_rul,
            "status": _rul_status(lstm_rul),
            "capped": bool(lstm_rul >= 124 or gbr_rul >= 124),
        })

    trajectory_cache[cache_key] = points
    return {
        "dataset_name": request.dataset_name,
        "unit_number": request.unit_number,
        "points": points,
    }


class ExplainLSTMRequest(BaseModel):
    dataset_name: str
    unit_number: int
    readings: List[SensorReading]


@app.post("/explain/lstm")
def explain_lstm(request: ExplainLSTMRequest):
    if request.dataset_name not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {request.dataset_name}")
    if not request.readings:
        raise HTTPException(status_code=400, detail="readings must not be empty")

    ds_artifacts = artifacts[request.dataset_name]
    df = pd.DataFrame([r.model_dump() for r in request.readings])
    df["unit_number"] = request.unit_number

    df_proc = assign_regimes(df, ds_artifacts["kmeans_lstm"])
    all_sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    df_proc = normalize_by_regime(df_proc, ds_artifacts["norm_stats"], all_sensor_cols)

    inf_sensors = ds_artifacts["informative_sensors_lstm"]
    X_base, _ = build_test_sequences(df_proc, inf_sensors, window_length=30)
    lstm_model = ds_artifacts["lstm_model"]
    base_rul = float(lstm_model.predict(X_base, verbose=0).flatten()[0])

    # 6 bins of 5 cycles each across the 30-cycle window
    cycle_bins = ["t-29..t-25", "t-24..t-20", "t-19..t-15", "t-14..t-10", "t-9..t-5", "t-4..t"]
    bin_size = 5
    occ_batch = []
    coords = []
    for b_idx in range(len(cycle_bins)):
        start = b_idx * bin_size
        end = start + bin_size
        for s_idx in range(len(inf_sensors)):
            perturbed = X_base[0].copy()
            perturbed[start:end, s_idx] = 0.0
            occ_batch.append(perturbed)
            coords.append((b_idx, s_idx))

    occ_preds = lstm_model.predict(np.array(occ_batch), verbose=0).flatten()
    matrix = [
        [b_idx, s_idx, round(float(abs(pred - base_rul)), 3)]
        for (b_idx, s_idx), pred in zip(coords, occ_preds)
    ]

    return {
        "dataset_name": request.dataset_name,
        "unit_number": request.unit_number,
        "base_rul": round(base_rul, 2),
        "sensors": inf_sensors,
        "cycle_bins": cycle_bins,
        "matrix": matrix,
    }


class BatchReading(SensorReading):
    unit_number: int


class BatchPredictRequest(BaseModel):
    dataset_name: str
    rows: List[BatchReading]


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    if request.dataset_name not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {request.dataset_name}")
    if not request.rows:
        raise HTTPException(status_code=400, detail="rows must not be empty")

    df = pd.DataFrame([r.model_dump() for r in request.rows])
    ds_artifacts = artifacts[request.dataset_name]

    df_gbr = assign_regimes(df, ds_artifacts["kmeans_gbr"])
    df_gbr = build_baseline_features(df_gbr, ds_artifacts["informative_sensors_gbr"])
    last_rows = get_last_cycle_per_unit(df_gbr).sort_values("unit_number")
    gbr_preds = ds_artifacts["gbr_model"].predict(last_rows[ds_artifacts["feature_cols"]])
    gbr_by_unit = {int(uid): float(p) for uid, p in zip(last_rows["unit_number"], gbr_preds)}

    df_lstm = assign_regimes(df, ds_artifacts["kmeans_lstm"])
    all_sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    df_lstm = normalize_by_regime(df_lstm, ds_artifacts["norm_stats"], all_sensor_cols)
    X_seq, unit_ids = build_test_sequences(
        df_lstm, ds_artifacts["informative_sensors_lstm"], window_length=30
    )
    lstm_preds = ds_artifacts["lstm_model"].predict(X_seq, verbose=0).flatten()
    lstm_by_unit = {int(uid): float(p) for uid, p in zip(unit_ids, lstm_preds)}

    results = []
    for _, row in last_rows.iterrows():
        uid = int(row["unit_number"])
        gbr_rul = round(gbr_by_unit[uid], 2)
        lstm_rul = round(lstm_by_unit[uid], 2)
        results.append({
            "unit_number": uid,
            "total_cycles": int(row["time_cycles"]),
            "gbr_rul": gbr_rul,
            "lstm_rul": lstm_rul,
            "status": _rul_status(lstm_rul),
            "capped": bool(lstm_rul >= 124 or gbr_rul >= 124),
        })

    return {"dataset_name": request.dataset_name, "results": results}


@app.get("/metrics/breakdown")
def get_metrics_breakdown():
    global breakdown_cache
    if breakdown_cache is not None:
        return {"breakdown": breakdown_cache}

    rows = []
    for ds in DATASETS:
        engines = list_dataset_engines(ds)["engines"]
        with_rul = [e for e in engines if e["true_rul"] is not None]
        within_cap = [e for e in with_rul if e["true_rul"] <= 125]

        for model_name, key in [("GBR", "gbr_rul"), ("LSTM", "lstm_rul")]:
            all_m = evaluate([e["true_rul"] for e in with_rul], [e[key] for e in with_rul])
            cap_m = evaluate([e["true_rul"] for e in within_cap], [e[key] for e in within_cap])
            rows.append({
                "dataset": ds,
                "model": model_name,
                "n_all": len(with_rul),
                "n_within_cap": len(within_cap),
                "RMSE_all": round(float(all_m["RMSE"]), 2),
                "PHM08_all": round(float(all_m["PHM08_score"]), 2),
                "RMSE_within_cap": round(float(cap_m["RMSE"]), 2),
                "PHM08_within_cap": round(float(cap_m["PHM08_score"]), 2),
            })

    breakdown_cache = rows
    return {"breakdown": rows}


FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")