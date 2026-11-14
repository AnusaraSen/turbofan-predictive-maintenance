import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from api.main import app
from feature_engineering import build_baseline_features
from preprocessing import (
    assign_regimes,
    compute_rul,
    fit_regime_clusters,
    fit_regime_norm_stats,
    normalize_by_regime,
    select_informative_sensors,
)
from scoring import evaluate, phm08_per_engine, phm08_score, rmse
from sequence_builder import (
    build_test_sequences,
    build_train_sequences,
    simulate_test_sequences,
)


def _make_synthetic_engine_df(unit_lengths: dict[int, int], n_regimes: int = 1) -> pd.DataFrame:
    """Create synthetic C-MAPSS-like engine trajectories for unit tests."""
    rng = np.random.default_rng(123)
    rows = []
    for unit_id, length in unit_lengths.items():
        for cycle in range(1, length + 1):
            regime = (cycle - 1) % n_regimes
            op1 = float(regime * 20.0)
            op2 = float(regime * 0.25)
            op3 = float(100.0 if regime % 2 == 0 else 60.0)
            # sensor_1: constant within each regime, shifts across regimes (uninformative)
            s1 = 500.0 + regime * 50.0
            # sensor_2: degrades with cycle + small noise (informative)
            s2 = 600.0 + 0.5 * cycle + float(rng.normal(0, 0.1))
            # sensor_3: constant globally (uninformative)
            s3 = 1400.0
            rows.append({
                "unit_number": unit_id,
                "time_cycles": cycle,
                "op_setting_1": op1,
                "op_setting_2": op2,
                "op_setting_3": op3,
                "sensor_1": s1,
                "sensor_2": s2,
                "sensor_3": s3,
            })
    return pd.DataFrame(rows)


def _make_api_readings(n_cycles: int = 30) -> list[dict]:
    """Create valid SensorReading payloads for FastAPI /predict tests."""
    readings = []
    for cycle in range(1, n_cycles + 1):
        row = {
            "time_cycles": cycle,
            "op_setting_1": 0.001,
            "op_setting_2": 0.0002,
            "op_setting_3": 100.0,
        }
        for i in range(1, 22):
            row[f"sensor_{i}"] = 500.0 + i + 0.1 * cycle
        readings.append(row)
    return readings


# --- Scoring tests -----------------------------------------------------------

def test_rmse_and_phm08_worked_examples():
    assert rmse([10, 20], [13, 16]) == pytest.approx(3.5355339, rel=1e-5)

    # Worked examples from scoring.py: d=-10->1.16, d=+10->1.72, d=-30->9.05, d=+30->19.09
    assert phm08_score([100], [90]) == pytest.approx(np.exp(10 / 13) - 1)
    assert phm08_score([100], [110]) == pytest.approx(np.exp(10 / 10) - 1)
    assert phm08_score([100], [70]) == pytest.approx(np.exp(30 / 13) - 1)
    assert phm08_score([100], [130]) == pytest.approx(np.exp(30 / 10) - 1)

    # Asymmetric penalty: overprediction (+d) > underprediction (-d)
    assert phm08_score([100], [120]) > phm08_score([100], [80])


def test_evaluate_and_per_engine_breakdown():
    y_true = [50, 100]
    y_pred = [40, 110]
    metrics = evaluate(y_true, y_pred)
    assert set(metrics.keys()) == {"RMSE", "PHM08_score"}
    assert metrics["RMSE"] == pytest.approx(10.0)

    per_engine = phm08_per_engine(y_true, y_pred, unit_ids=[1, 2])
    assert list(per_engine["unit_id"]) == [1, 2]
    assert list(per_engine["d"]) == [-10.0, 10.0]
    assert per_engine["phm08_contribution"].sum() == pytest.approx(metrics["PHM08_score"])


# --- Preprocessing tests -----------------------------------------------------

def test_compute_rul_piecewise_cap():
    df = _make_synthetic_engine_df({1: 150, 2: 40})
    labeled = compute_rul(df, rul_cap=125)

    u1 = labeled[labeled["unit_number"] == 1].sort_values("time_cycles")
    assert u1["RUL"].iloc[0] == 125
    assert u1["RUL"].iloc[-1] == 0
    assert u1["RUL"].max() == 125

    u2 = labeled[labeled["unit_number"] == 2].sort_values("time_cycles")
    assert u2["RUL"].iloc[0] == 39
    assert u2["RUL"].iloc[-1] == 0


def test_regime_clustering_normalization_and_sensor_selection():
    df = _make_synthetic_engine_df({1: 60, 2: 60}, n_regimes=6)
    kmeans = fit_regime_clusters(df, n_regimes=6, random_seed=42)
    df = assign_regimes(df, kmeans)
    assert df["regime_id"].nunique() == 6

    sensors = ["sensor_1", "sensor_2", "sensor_3"]
    # sensor_1 shifts across regimes but is constant within each regime -> must be dropped
    informative, dropped = select_informative_sensors(df, sensors, variance_threshold=0.01)
    assert informative == ["sensor_2"]
    assert set(dropped) == {"sensor_1", "sensor_3"}

    norm_stats = fit_regime_norm_stats(df, sensors)
    norm_df = normalize_by_regime(df, norm_stats, sensors)
    assert not norm_df[sensors].isna().any().any()
    for _, grp in norm_df.groupby("regime_id"):
        assert grp["sensor_2"].mean() == pytest.approx(0.0, abs=1e-6)


# --- Sequence builder tests --------------------------------------------------

def test_train_and_test_sequence_shapes_and_padding():
    df = _make_synthetic_engine_df({1: 40, 2: 12})
    df = compute_rul(df, rul_cap=125)
    sensors = ["sensor_1", "sensor_2"]

    # Training skips unit 2 (length 12 < window 30), keeps 40 - 30 + 1 = 11 windows for unit 1
    X_train, y_train = build_train_sequences(df, sensors, window_length=30)
    assert X_train.shape == (11, 30, 2)
    assert y_train.shape == (11,)
    assert y_train[-1] == 0

    # Test pads short engine (unit 2) with zeros at the front
    X_test, unit_ids = build_test_sequences(df, sensors, window_length=30)
    assert X_test.shape == (2, 30, 2)
    assert list(unit_ids) == [1, 2]
    assert np.allclose(X_test[1, :18, :], 0.0)
    assert not np.allclose(X_test[1, 18:, :], 0.0)


def test_simulate_test_sequences_short_engine_edge_case():
    # Edge case: very short engines should not fail or wrap negative indices
    df = _make_synthetic_engine_df({1: 5, 2: 15, 3: 50})
    df = compute_rul(df, rul_cap=125)
    X_sim, y_sim, unit_ids = simulate_test_sequences(
        df, ["sensor_1", "sensor_2"], window_length=30, random_seed=42
    )
    assert X_sim.shape == (3, 30, 2)
    assert y_sim.shape == (3,)
    assert list(unit_ids) == [1, 2, 3]
    assert not np.isnan(X_sim).any()


# --- Feature engineering tests -----------------------------------------------

def test_feature_engineering_no_nans_and_no_cross_engine_leakage():
    df = _make_synthetic_engine_df({1: 15, 2: 15})
    # Shift unit 2 sensor_2 far from unit 1 to detect any cross-engine leakage
    df.loc[df["unit_number"] == 2, "sensor_2"] += 1000.0

    feat_df = build_baseline_features(df, ["sensor_2"])
    expected_cols = {
        "sensor_2_rolling_mean_10",
        "sensor_2_rolling_std_10",
        "sensor_2_rate_of_change_1",
        "sensor_2_lag_1",
    }
    assert expected_cols.issubset(feat_df.columns)
    assert not feat_df[list(expected_cols)].isna().any().any()

    u2_first = feat_df[feat_df["unit_number"] == 2].sort_values("time_cycles").iloc[0]
    assert u2_first["sensor_2_rate_of_change_1"] == 0.0
    assert u2_first["sensor_2_lag_1"] == pytest.approx(u2_first["sensor_2"])
    assert u2_first["sensor_2_rolling_mean_10"] == pytest.approx(u2_first["sensor_2"])


# --- FastAPI endpoint smoke tests --------------------------------------------

def test_api_predict_and_feature_importance():
    client = TestClient(app)
    readings = _make_api_readings(30)

    for model_type in ("gbr", "lstm"):
        resp = client.post(
            "/predict",
            json={
                "dataset_name": "FD001",
                "model_type": model_type,
                "unit_number": 1,
                "readings": readings,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["dataset_name"] == "FD001"
        assert body["model_type"] == model_type
        assert isinstance(body["predicted_RUL"], float)
        assert isinstance(body["capped"], bool)

    bad_ds = client.post(
        "/predict",
        json={"dataset_name": "FD999", "model_type": "gbr", "unit_number": 1, "readings": readings},
    )
    assert bad_ds.status_code == 400

    bad_model = client.post(
        "/predict",
        json={"dataset_name": "FD001", "model_type": "xgb", "unit_number": 1, "readings": readings},
    )
    assert bad_model.status_code == 400

    fi_resp = client.get("/feature-importance/FD001")
    assert fi_resp.status_code == 200
    assert len(fi_resp.json()["features"]) > 0

    assert client.get("/feature-importance/FD999").status_code == 400


def test_api_phase1_fleet_and_metrics_endpoints():
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert len(metrics.json()["metrics"]) == 8

    fleet = client.get("/datasets/FD001/engines")
    assert fleet.status_code == 200
    engines = fleet.json()["engines"]
    assert len(engines) == 100
    assert {"unit_number", "total_cycles", "gbr_rul", "lstm_rul", "status", "capped"}.issubset(
        engines[0].keys()
    )
    assert isinstance(engines[0]["true_rul"], float)

    detail = client.get("/datasets/FD001/engines/1")
    assert detail.status_code == 200
    body = detail.json()
    assert body["unit_number"] == 1
    assert len(body["raw_rows"]) == body["total_cycles"]
    assert "regime_id" in body["raw_rows"][0]

    assert client.get("/datasets/FD001/engines/99999").status_code == 404

    traj = client.post("/predict/trajectory", json={"dataset_name": "FD001", "unit_number": 1})
    assert traj.status_code == 200
    pts = traj.json()["points"]
    assert len(pts) == body["total_cycles"]
    assert {"time_cycles", "regime_id", "gbr_rul", "lstm_rul", "status", "capped"}.issubset(
        pts[0].keys()
    )
    assert isinstance(pts[-1]["true_rul"], float)
    assert pts[0]["true_rul"] > pts[-1]["true_rul"]

    # Phase 3 endpoints: /explain/lstm, /predict/batch, /metrics/breakdown
    readings = _make_api_readings(30)
    explain = client.post(
        "/explain/lstm",
        json={"dataset_name": "FD001", "unit_number": 1, "readings": readings},
    )
    assert explain.status_code == 200
    exp_body = explain.json()
    assert len(exp_body["cycle_bins"]) == 6
    assert len(exp_body["matrix"]) == 6 * len(exp_body["sensors"])

    batch_rows = [{**r, "unit_number": 1} for r in readings] + [
        {**r, "unit_number": 2} for r in readings
    ]
    batch = client.post(
        "/predict/batch",
        json={"dataset_name": "FD001", "rows": batch_rows},
    )
    assert batch.status_code == 200
    assert len(batch.json()["results"]) == 2

    breakdown = client.get("/metrics/breakdown")
    assert breakdown.status_code == 200
    assert len(breakdown.json()["breakdown"]) == 8
