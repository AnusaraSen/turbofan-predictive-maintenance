import numpy as np
from data_loader import load_train_data
from preprocessing import assign_regimes, fit_regime_clusters, compute_rul, select_informative_sensors
from feature_engineering import build_baseline_features

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

if __name__ == "__main__":
    
    train_df = load_train_data("FD001")
    train_df = compute_rul(train_df)
    
    kmeans = fit_regime_clusters(train_df, n_regimes=1)
    train_df = assign_regimes(train_df, kmeans)

    all_sensor_columns = [col for col in train_df.columns if col.startswith("sensor")]
    informative_sensors, dropped_sensors = select_informative_sensors(train_df, all_sensor_columns)

    featured_df = build_baseline_features(train_df, informative_sensors)
    train_df, val_df = split_by_unit(featured_df, test_size=0.2, random_seed=42)

    print(f"Train set shape: {train_df.shape}, Validation set shape: {val_df.shape}")