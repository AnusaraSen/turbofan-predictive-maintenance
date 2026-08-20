import pandas as pd
from sklearn.cluster import KMeans
from data_loader import load_train_data  

"""
1. max_cycles = df.groupby("unit_number")["time_cycles"].max()

Each unit_number is one engine's full run-to-failure history — a sequence of rows, one per operating cycle, until it fails. Grouping by unit_number and taking .max() on time_cycles gives you the last recorded cycle for each engine — i.e., the cycle number at which that engine failed. This returns a Series indexed by unit_number.

2. df.merge(max_cycles.reset_index(name="max_cycle"), on="unit_number")

max_cycles is a Series; .reset_index(name="max_cycle") turns it into a two-column DataFrame (unit_number, max_cycle) so it can be merged back onto the original data. The merge broadcasts each engine's failure cycle onto every row belonging to that engine — so a row at cycle 50 for unit 3 now knows unit 3 ultimately failed at, say, cycle 200.

3. df["RUL"] = df["max_cycle"] - df["time_cycles"]

This is the actual label: how many cycles remain before failure, at each point in time. Row at cycle 50 with max_cycle 200 → RUL = 150. As cycles increase toward max_cycle, RUL counts down to 0 at the final row (the failure point).

4. df["RUL"] = df["RUL"].clip(upper=rul_cap)

This caps any RUL value above 125 down to 125. .clip(upper=...) leaves values below the cap untouched and truncates everything above it.

Why cap RUL at all? This is the important modeling decision, not just cleanup. Early in an engine's life (e.g., cycle 5 of 200), degradation hasn't really started — sensors look nearly identical to a healthy new engine. If you let the model try to regress on the true linear RUL (195, 194, 193...) 
during this healthy phase, you're asking it to predict something the sensor data has no signal for yet, which just adds noise and hurts training. Capping RUL says: "beyond ~125 cycles out, just call it 125 — healthy is healthy." This turns the target into a piecewise-linear curve (flat at 125, then linearly decreasing near end-of-life) which matches the physical reality that degradation is only detectable in the later portion of an engine's life. This is the standard approach from the original C-MAPSS RUL papers (Heimes 2008, Saxena et al.), and 125 is the commonly used cap value.

"""


def compute_rul(train_df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    df = train_df.copy()
    # TODO: get max cycle per unit_number

    max_cycles = df.groupby("unit_number")["time_cycles"].max()
    df = df.merge(max_cycles.reset_index(name="max_cycle"), on="unit_number")

    # TODO: compute raw RUL = max_cycle - current cycle
    df["RUL"] = df["max_cycle"] - df["time_cycles"]
    # TODO: cap it at rul_cap
    df["RUL"] = df["RUL"].clip(upper=rul_cap)
    return df

def fit_regime_clusters(df: pd.DataFrame, n_regimes: int, random_seed: int = 42) -> KMeans:
    op_setting_columns = ["op_setting_1", "op_setting_2", "op_setting_3"]
    # TODO: create a KMeans model with n_clusters=n_regimes
    kmeans = KMeans(n_clusters=n_regimes, random_state=random_seed)

    # TODO: fit it on df[op_setting_columns]
    kmeans.fit(df[op_setting_columns])
    return kmeans


def assign_regimes(df: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    df = df.copy()
    op_setting_columns = ["op_setting_1", "op_setting_2", "op_setting_3"]
    # TODO: use kmeans.predict() to assign a regime_id column
    df["regime_id"] = kmeans.predict(df[op_setting_columns])
    return df


def fit_regime_norm_stats(df: pd.DataFrame, sensor_columns: list) -> pd.DataFrame:
    # TODO: group by regime_id, compute mean and std for each sensor
    stats = df.groupby("regime_id")[sensor_columns].agg(["mean", "std"])
    
    std_cols = stats.columns[stats.columns.get_level_values(1) == "std"]
    stats[std_cols] = stats[std_cols].replace(0, 1e-6)
    
    return stats


def normalize_by_regime(df: pd.DataFrame, norm_stats: pd.DataFrame, sensor_columns: list) -> pd.DataFrame:
    df = df.copy()
    # TODO: for each sensor, look up this row's regime's mean/std, apply z-score
    for col in sensor_columns:
        mean_col = (col, "mean")
        std_col = (col, "std")
        df[col] = (df[col] - df["regime_id"].map(norm_stats[mean_col])) / df["regime_id"].map(norm_stats[std_col])
    # this is the trickiest part - think about how to map each row's regime_id
    # to the right mean/std from norm_stats without a slow row-by-row loop
    return df

def select_informative_sensors(normalized_train_df: pd.DataFrame, sensor_columns: list, variance_threshold: float = 0.01):
    variances_by_regime = normalized_train_df.groupby("regime_id")[sensor_columns].var()
    sensor_variances = variances_by_regime.mean(axis=0)

    informative_sensors = sensor_variances[sensor_variances > variance_threshold].index.tolist()
    dropped_sensors = sensor_variances[sensor_variances <= variance_threshold].index.tolist()

    return informative_sensors, dropped_sensors

if __name__ == "__main__":
    

    train_df = load_train_data("FD002")
    train_df = compute_rul(train_df)
    """
    kmeans = fit_regime_clusters(train_df, n_regimes=1)
    train_df = assign_regimes(train_df, kmeans)

    sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

    # sensor selection on RAW values, before normalization
    informative_sensors, dropped_sensors = select_informative_sensors(train_df, sensor_columns)
    print("Informative sensors:", informative_sensors)
    print("Dropped sensors:", dropped_sensors)

    # normalization still happens, just after selection, and isn't what selection is based on
    norm_stats = fit_regime_norm_stats(train_df, sensor_columns)
    normalized_train_df = normalize_by_regime(train_df, norm_stats, sensor_columns)
    """
    print(train_df[["op_setting_1", "op_setting_2", "op_setting_3"]].describe())

    # rough check: round and count unique combinations
    rounded = train_df[["op_setting_1", "op_setting_2"]].round(1)
    print(rounded.drop_duplicates().shape)
     
    kmeans = fit_regime_clusters(train_df, n_regimes=6)
    train_df = assign_regimes(train_df, kmeans)

    print(train_df["regime_id"].value_counts().sort_index())


   