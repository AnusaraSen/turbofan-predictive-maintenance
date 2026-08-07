from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

INDEX_COLUMNS = ["unit_number", "time_cycles"]
OP_SETTING_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

ALL_COLUMNS = INDEX_COLUMNS + OP_SETTING_COLUMNS + SENSOR_COLUMNS