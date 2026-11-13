import joblib
from pathlib import Path
import keras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import PROJECT_ROOT
from feature_engineering import build_baseline_features
from preprocessing import assign_regimes, normalize_by_regime
from sequence_builder import build_test_sequences
from train_baseline import get_last_cycle_per_unit


app = FastAPI(title="Turbofan RUL Prediction API")

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
artifacts = {} 

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