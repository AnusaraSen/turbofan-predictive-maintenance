from data_loader import load_test_data, load_rul_data
from config import PROJECT_ROOT

for dataset_name in ["FD001", "FD002", "FD003", "FD004"]:
    test_df = load_test_data(dataset_name)
    rul_df = load_rul_data(dataset_name)  # columns: RUL, unit_number

    # RUL_FDxxx.txt gives the true RUL at each engine's LAST recorded cycle -
    # merge it onto every row for that engine (simplest for the dashboard to use,
    # since it just needs "this engine's true RUL" without recomputing per-row RUL)
    sample_df = test_df.merge(rul_df, on="unit_number", how="left")

    dataset_dir = PROJECT_ROOT / "models" / dataset_name
    sample_df.to_csv(dataset_dir / "sample_engines.csv", index=False)
    print(f"{dataset_name}: saved {sample_df['unit_number'].nunique()} engines, "
          f"RUL column present: {'RUL' in sample_df.columns}")