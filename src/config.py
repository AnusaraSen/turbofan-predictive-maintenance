from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

INDEX_COLUMNS = ["unit_number", "time_cycles"]
OP_SETTING_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

ALL_COLUMNS = INDEX_COLUMNS + OP_SETTING_COLUMNS + SENSOR_COLUMNS

DATASET_CONFIG = {
    "FD001": {"n_regimes": 1, "n_fault_modes": 1, "train_units": 100, "test_units": 100},
    "FD002": {"n_regimes": 6, "n_fault_modes": 1, "train_units": 260, "test_units": 259},
    "FD003": {"n_regimes": 1, "n_fault_modes": 2, "train_units": 100, "test_units": 100},
    "FD004": {"n_regimes": 6, "n_fault_modes": 2, "train_units": 248, "test_units": 249},
}