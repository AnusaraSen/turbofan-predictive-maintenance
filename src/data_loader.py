from os import path

import pandas as pd
from config import RAW_DATA_DIR, ALL_COLUMNS


def load_train_data(dataset_name: str = "FD001") -> pd.DataFrame:
    """
    Load the raw train_FDxxx.txt file into a DataFrame with proper column names.
    """
    path = RAW_DATA_DIR / f"train_{dataset_name}.txt"

    df = pd.read_csv(path, header=None, sep=r'\s+')
    print(df.shape) 
    
    if df.shape[1] == len(ALL_COLUMNS):
        df.columns = ALL_COLUMNS

    return df

def load_test_data(dataset_name: str = "FD001") -> pd.DataFrame:
    """
    Load the raw test_FDxxx.txt file — same structure as train, but
    trajectories are truncated before failure.
    """
    path = RAW_DATA_DIR / f"test_{dataset_name}.txt"
    
    df = pd.read_csv(path, header=None, sep=r'\s+')
    print(df.shape) 
    
    if df.shape[1] == len(ALL_COLUMNS):
        df.columns = ALL_COLUMNS
    else:
        raise ValueError(
            f"Expected {len(ALL_COLUMNS)} columns, got {df.shape[1]} for {path}"
        )

    return df

def load_rul_data(dataset_name: str = "FD001") -> pd.DataFrame:
    """
    Load the RUL_FDxxx.txt file — the true RUL for each engine in the test set.
    """
    path = RAW_DATA_DIR / f"RUL_{dataset_name}.txt"

    # TODO: read the file (single column, no header)
    df = pd.read_csv(path, header=None)

    # TODO: add a unit_number column based on row position (starting at 1)
    df['unit_number'] = df.index + 1
    df.rename(columns={0: 'RUL'}, inplace=True)   

    return df

if __name__ == "__main__":

    """
    train_df = load_train_data("FD001")
    test_df = load_test_data("FD001")
    rul_df = load_rul_data("FD001")

    assert train_df["unit_number"].nunique() == 100, "Expected 100 engines in FD001 train set"
    assert test_df["unit_number"].nunique() == 100, "Expected 100 engines in FD001 test set"
    assert len(rul_df) == 100, "Expected 100 RUL values in FD001"
    print("All checks passed.")
    """
    print("Loading FD002 data...")

    train_df = load_train_data("FD002")
    test_df = load_test_data("FD002")
    rul_df = load_rul_data("FD002")

    

    print(train_df.shape)
    print(train_df["unit_number"].nunique())  # README says 260 train units