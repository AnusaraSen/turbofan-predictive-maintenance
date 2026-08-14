import numpy as np

def build_train_sequences(df, sensor_columns, window_length):
    X, y = [], []
    for unit_id, group in df.groupby("unit_number"):
        group = group.sort_values("time_cycles")
        sensor_values = group[sensor_columns].values
        rul_values = group["RUL"].values
        n_cycles = len(group)

        if n_cycles < window_length:
            continue  # left as skip for training, per our earlier decision - not the padding fix

        for start in range(n_cycles - window_length + 1):
            window = sensor_values[start:start + window_length]
            label = rul_values[start + window_length - 1]
            X.append(window)
            y.append(label)

    return np.array(X), np.array(y)


def build_test_sequences(df, sensor_columns, window_length):
    X, unit_ids = [], []
    for unit_id, group in df.groupby("unit_number"):
        group = group.sort_values("time_cycles")
        sensor_values = group[sensor_columns].values
        n_cycles = len(group)

        if n_cycles >= window_length:
            window = sensor_values[-window_length:]
        else:
            pad_width = window_length - n_cycles
            padding = np.zeros((pad_width, sensor_values.shape[1]))
            window = np.vstack([padding, sensor_values])

        X.append(window)
        unit_ids.append(unit_id)

    return np.array(X), np.array(unit_ids)

def simulate_test_sequences(df, sensor_columns, window_length, random_seed=42):
    rng = np.random.default_rng(random_seed)
    X, y, unit_ids = [], [], []

    for unit_id, group in df.groupby("unit_number"):
        group = group.sort_values("time_cycles")
        sensor_values = group[sensor_columns].values
        rul_values = group["RUL"].values
        n_cycles = len(group)

        cutoff_idx = rng.integers(0, n_cycles)

        # how many real cycles exist up to and including the cutoff
        available = cutoff_idx + 1

        if available >= window_length:
            window = sensor_values[cutoff_idx - window_length + 1 : cutoff_idx + 1]
        else:
            pad_width = window_length - available
            padding = np.zeros((pad_width, sensor_values.shape[1]))
            window = np.vstack([padding, sensor_values[:cutoff_idx + 1]])

        label = rul_values[cutoff_idx]

        X.append(window)
        y.append(label)
        unit_ids.append(unit_id)

    return np.array(X), np.array(y), np.array(unit_ids)