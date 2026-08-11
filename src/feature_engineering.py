import pandas as pd
from data_loader import load_train_data
from preprocessing import compute_rul, fit_regime_clusters, assign_regimes, select_informative_sensors


def add_rolling_features(df: pd.DataFrame, sensor_columns: list, windows: tuple = (5, 10, 20)) -> pd.DataFrame:
    df = df.sort_values(["unit_number", "time_cycles"]).copy()
    grouped = df.groupby("unit_number")

    for w in windows:
        for sensor in sensor_columns:
            # TODO: rolling mean per engine, window size w
            rolling_mean = grouped[sensor].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"{sensor}_rolling_mean_{w}"] = rolling_mean
            

            # TODO: rolling std per engine, window size w (watch for NaN on the first row or two - fillna(0) is reasonable)
            rolling_std = grouped[sensor].rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True)
            df[f"{sensor}_rolling_std_{w}"] = rolling_std.fillna(0)

    return df


def add_rate_of_change(df: pd.DataFrame, sensor_columns: list, lag: int = 1) -> pd.DataFrame:
    df = df.sort_values(["unit_number", "time_cycles"]).copy()
    grouped = df.groupby("unit_number")

    for sensor in sensor_columns:
        # TODO: current value minus value `lag` cycles ago, within each engine
        # hint: .diff(periods=lag) inside a groupby - think about what the very first
        # row of each engine should get, since there's no prior value to diff against
        df[f"{sensor}_rate_of_change_{lag}"] = grouped[sensor].diff(periods=lag).fillna(0)
    return df


def add_lag_features(df: pd.DataFrame, sensor_columns: list, lags: tuple = (1, 3, 5)) -> pd.DataFrame:
    df = df.sort_values(["unit_number", "time_cycles"]).copy()
    grouped = df.groupby("unit_number")

    for lag in lags:
        for sensor in sensor_columns:
            lagged_values = grouped[sensor].shift(lag)
            col_name = f"{sensor}_lag_{lag}"
            df[col_name] = lagged_values
            # backfill within each engine so the first `lag` rows get the
            # earliest available real value, instead of NaN
            df[col_name] = df.groupby("unit_number")[col_name].bfill()
    return df

def build_baseline_features(
    df: pd.DataFrame,
    sensor_columns: list,
    rolling_windows: tuple = (10,),   # NARROW SET for first pass - originally designed for (5, 10, 20)
    lags: tuple = (1,),                # NARROW SET for first pass - originally designed for (1, 3, 5)
) -> pd.DataFrame:
    """
    NOTE: started narrow (single window=10, single lag=1) to get a fast,
    interpretable first baseline. The full engineered feature set supports
    windows=(5,10,20) and lags=(1,3,5) - revisit widening this AFTER
    reviewing feature importances from the first trained model, not before.
    """
    df = add_rolling_features(df, sensor_columns, windows=rolling_windows)
    df = add_rate_of_change(df, sensor_columns)
    df = add_lag_features(df, sensor_columns, lags=lags)
    return df.copy()  # de-fragment, addresses the earlier PerformanceWarning too


if __name__ == "__main__":
    train_df = load_train_data("FD001")
    train_df = compute_rul(train_df)

    kmeans = fit_regime_clusters(train_df, n_regimes=1)
    train_df = assign_regimes(train_df, kmeans)

    all_sensor_columns = [col for col in train_df.columns if col.startswith("sensor")]
    informative_sensors, dropped_sensors = select_informative_sensors(train_df, all_sensor_columns)
    print("Using", len(informative_sensors), "informative sensors, dropped:", dropped_sensors)

    featured_df = build_baseline_features(train_df, informative_sensors)
    print(featured_df.shape)