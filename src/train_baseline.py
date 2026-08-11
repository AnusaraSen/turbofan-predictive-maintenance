import numpy as np

def split_by_unit(df, test_size=0.2, random_seed=42):
    unit_ids = df["unit_number"].unique()
    rng = np.random.default_rng(random_seed)
    shuffled = rng.permutation(unit_ids)
    n_val = int(len(unit_ids) * test_size)
    val_units = set(shuffled[:n_val])
    train_units = set(shuffled[n_val:])

    train_df = df[df["unit_number"].isin(train_units)]
    val_df = df[df["unit_number"].isin(val_units)]
    return train_df, val_df