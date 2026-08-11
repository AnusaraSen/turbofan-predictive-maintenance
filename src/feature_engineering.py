import pandas as pd
from data_loader import load_train_data  

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

if __name__ == "__main__":
    train_df = load_train_data("FD001")
    sensor_columns = [col for col in train_df.columns if col.startswith("sensor")]
    
    train_df = add_rolling_features(train_df, sensor_columns)
    train_df = add_rate_of_change(train_df, sensor_columns)
    train_df = add_lag_features(train_df, sensor_columns)
    
    print(train_df[["unit_number", "time_cycles", "sensor_3_lag_1", ]].head(10))