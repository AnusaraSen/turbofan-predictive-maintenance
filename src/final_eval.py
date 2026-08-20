from data_loader import load_test_data, load_rul_data, load_train_data
from preprocessing import assign_regimes, fit_regime_norm_stats, normalize_by_regime, fit_regime_clusters
from feature_engineering import build_baseline_features
from sequence_builder import build_test_sequences
from scoring import evaluate, phm08_per_engine
from train_baseline import get_last_cycle_per_unit, run_baseline_training
import pandas as pd

from train_lstm import train_lstm_model

def evaluate_gbr_on_real_test(gbr_model, kmeans, informative_sensors, feature_cols, dataset_name="FD001"):
    test_df = load_test_data(dataset_name)
    rul_df = load_rul_data(dataset_name)

    # TODO: assign_regimes using the ALREADY-FITTED kmeans (don't refit)
    test_df = assign_regimes(test_df, kmeans)
    # TODO: build_baseline_features on informative_sensors (same as training)
    test_df = build_baseline_features(test_df, informative_sensors)
    # TODO: get the LAST row per engine (test set is already truncated -
    #        last row IS the prediction point - reuse your get_last_cycle_per_unit)
    test_df = get_last_cycle_per_unit(test_df)
    # TODO: predict RUL using gbr_model
    X_test = test_df[feature_cols]
    test_df["predicted_RUL"] = gbr_model.predict(X_test)
    # TODO: merge predictions with rul_df on unit_number (NOT row order)
    gbr_merged_df = test_df.merge(rul_df, on="unit_number", how="left")
    # TODO: evaluate(true_RUL, predicted_RUL), and phm08_per_engine(...)    
    metrics = evaluate(gbr_merged_df["RUL"], gbr_merged_df["predicted_RUL"])
    y_true, y_pred, unit_ids = gbr_merged_df["RUL"], gbr_merged_df["predicted_RUL"], gbr_merged_df["unit_number"]
    per_engine_df = phm08_per_engine(y_true, y_pred, unit_ids)

    '''
    Print evaluation metrics
    
    print(merged_df["RUL"].isna().sum())  # should be 0, no missing true RULs

    print(merged_df["RUL"].describe())
    print("Test engines with true RUL > 125:", (merged_df["RUL"] > 125).sum())
    print(merged_df[merged_df["RUL"] > 125][["unit_number", "RUL", "predicted_RUL"]])
    '''

    gbr_capped_ok = gbr_merged_df[gbr_merged_df["RUL"] <= 125]
    gbr_capped_bad = gbr_merged_df[gbr_merged_df["RUL"] > 125]
    print("GBR, within cap (n={}):".format(len(gbr_capped_ok)), evaluate(gbr_capped_ok["RUL"], gbr_capped_ok["predicted_RUL"]))
    print("GBR, beyond cap (n={}):".format(len(gbr_capped_bad)), evaluate(gbr_capped_bad["RUL"], gbr_capped_bad["predicted_RUL"]))

    return metrics, per_engine_df


def evaluate_lstm_on_real_test(lstm_model, kmeans, norm_stats, informative_sensors, window_length=30, dataset_name="FD001"):
    test_df = load_test_data(dataset_name)
    rul_df = load_rul_data(dataset_name)

    # TODO: assign_regimes using the ALREADY-FITTED kmeans
    test_df = assign_regimes(test_df, kmeans)
    # TODO: normalize_by_regime using the ALREADY-FITTED norm_stats
    sensor_columns = [col for col in test_df.columns if col.startswith("sensor")]
    test_df = normalize_by_regime(test_df,norm_stats,sensor_columns)
    # TODO: build_test_sequences (already handles "last window per engine",
    #        including padding for short engines - you already validated this)
    X_test, unit_ids = build_test_sequences(test_df, informative_sensors, window_length)
    # TODO: predict with lstm_model, .flatten() the output
    preds = lstm_model.predict(X_test).flatten()
    # TODO: merge with rul_df on unit_number (build_test_sequences returns unit_ids
    #        in the array - use that to build a small DataFrame first, then merge)
    pred_df = pd.DataFrame({"unit_number": unit_ids, "predicted_RUL": preds})
    lstm_merged_df = pred_df.merge(rul_df, on="unit_number", how="left")
    # TODO: evaluate(...), phm08_per_engine(...)
    metrics = evaluate(lstm_merged_df["RUL"], lstm_merged_df["predicted_RUL"])
    y_true, y_pred, unit_ids = lstm_merged_df["RUL"], lstm_merged_df["predicted_RUL"], lstm_merged_df["unit_number"]
    per_engine_df = phm08_per_engine(y_true, y_pred, unit_ids)

    
    # LSTM
    lstm_capped_ok = lstm_merged_df[lstm_merged_df["RUL"] <= 125]
    lstm_capped_bad = lstm_merged_df[lstm_merged_df["RUL"] > 125]
    print("LSTM, within cap (n={}):".format(len(lstm_capped_ok)), evaluate(lstm_capped_ok["RUL"], lstm_capped_ok["predicted_RUL"]))
    print("LSTM, beyond cap (n={}):".format(len(lstm_capped_bad)), evaluate(lstm_capped_bad["RUL"], lstm_capped_bad["predicted_RUL"]))
        

    return metrics, per_engine_df

if __name__ == "__main__":
    dataset_name = "FD004"

    gbr_model, gbr_per_engine_val, gbr_metrics_val, kmeans_gbr, informative_sensors_gbr, feature_cols = run_baseline_training(dataset_name=dataset_name)

    gbr_test_metrics, gbr_test_per_engine = evaluate_gbr_on_real_test(
        gbr_model, kmeans_gbr, informative_sensors_gbr, feature_cols, dataset_name=dataset_name
    )
    print("Real test set results with GBR:")
    print(gbr_test_metrics)
    print(gbr_test_per_engine.sort_values("phm08_contribution", ascending=False))

    lstm_model, lstm_val_per_engine, lstm_val_metrics, kmeans_lstm, informative_sensors_lstm, history, norm_stats = train_lstm_model(dataset_name=dataset_name)

    lstm_test_metrics, lstm_test_per_engine = evaluate_lstm_on_real_test(
        lstm_model, kmeans_lstm, norm_stats, informative_sensors_lstm, window_length=30, dataset_name=dataset_name
    )

    

    print("LSTM internal validation results:", lstm_val_metrics)
    print("LSTM real test set results:", lstm_test_metrics)
    print(lstm_test_per_engine.sort_values("phm08_contribution", ascending=False))

    