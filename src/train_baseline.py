import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from data_loader import load_train_data
from preprocessing import assign_regimes, fit_regime_clusters, compute_rul, select_informative_sensors
from feature_engineering import build_baseline_features
from scoring import evaluate, phm08_per_engine
from config import DATASET_CONFIG

def split_by_unit(df, test_size=0.2, random_seed=42):

    # Split the data by unit number
    unit_ids = df["unit_number"].unique()

    # Use a random seed for reproducibility (so we get the same train/val split each time)
    rng = np.random.default_rng(random_seed)

    # Shuffle the unit IDs and split into train/val sets
    shuffled = rng.permutation(unit_ids)

    # Determine the number of units to include in the validation set (20% of the total)
    n_val = int(len(unit_ids) * test_size)

    # Create sets of unit IDs for the validation and training sets
    val_units = set(shuffled[:n_val])
    train_units = set(shuffled[n_val:])

    # Filter the original DataFrame to create the training and validation sets based on unit IDs
    train_df = df[df["unit_number"].isin(train_units)]
    val_df = df[df["unit_number"].isin(val_units)]

    #confirm set(train_df.unit_number.unique()) & set(val_df.unit_number.unique()) is empty
    assert set(train_df.unit_number.unique()) & set(val_df.unit_number.unique()) == set()

    return train_df, val_df

def get_feature_columns(df):
    exclude = ["unit_number", "time_cycles", "RUL", "max_cycle", "regime_id",
               "op_setting_1", "op_setting_2", "op_setting_3"]
    
    return [col for col in df.columns if col not in exclude]

def get_last_cycle_per_unit(df):
    # TODO: for each unit_number, keep only the row with the max time_cycles
    # hint: think about groupby + idxmax, or sort + drop_duplicates(keep="last")
    last_cycle_df = df.loc[df.groupby("unit_number")["time_cycles"].idxmax()]
    return last_cycle_df

def simulate_test_truncation(df, random_seed=42):
    rng = np.random.default_rng(random_seed)
    truncated_rows = []

    for unit_id, group in df.groupby("unit_number"):
        group = group.sort_values("time_cycles")
        # TODO: pick a random cutoff cycle for this engine - somewhere between
        # cycle 1 and its last cycle (inclusive) - then keep only that one row

        n_cycles = len(group)
        cutoff_idx = rng.integers(0, n_cycles)  # pick a random integer index into `group`
        truncated_rows.append(group.iloc[[cutoff_idx]])

    return pd.concat(truncated_rows)



def run_baseline_training(dataset_name="FD004"):
    train_df = load_train_data(dataset_name)
    train_df = compute_rul(train_df)

    n_regimes = DATASET_CONFIG[dataset_name]["n_regimes"]
    kmeans = fit_regime_clusters(train_df, n_regimes=n_regimes)
    train_df = assign_regimes(train_df, kmeans)

    all_sensor_columns = [col for col in train_df.columns if col.startswith("sensor")]
    informative_sensors, dropped_sensors = select_informative_sensors(train_df, all_sensor_columns)

    featured_df = build_baseline_features(train_df, informative_sensors)
    train_df, val_df = split_by_unit(featured_df, test_size=0.2, random_seed=42)
    simulated_val_df = simulate_test_truncation(val_df, random_seed=42)

    feature_cols = get_feature_columns(train_df)
    X_train, y_train = train_df[feature_cols], train_df["RUL"]
    X_val, y_val = simulated_val_df[feature_cols], simulated_val_df["RUL"]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    per_engine_df = phm08_per_engine(y_val, preds, simulated_val_df["unit_number"])
    metrics = evaluate(y_val, preds)

    """
    print(train_df.shape)  # confirm (53759, 26) or however many columns you have now
    print(train_df["unit_number"].nunique())  # should be 260, not 100

    lifespans = train_df.groupby("unit_number")["time_cycles"].max()
    print("Shortest engine life:", lifespans.min())
    print("Engines shorter than 30:", (lifespans < 30).sum())
    print("Unique regimes assigned:", train_df["regime_id"].nunique())
    """

    # return everything you might want to inspect later, not just the model
    return model, per_engine_df, metrics, kmeans, informative_sensors, feature_cols

if __name__ == "__main__":
    model, per_engine_df, metrics, kmeans, informative_sensors, feature_cols = run_baseline_training()
    print(metrics)
    print(per_engine_df.sort_values("phm08_contribution", ascending=False))

