import pandas as pd
import json

df = pd.read_csv("../models/FD004/sample_engines.csv")
one_engine = df[df["unit_number"] == df["unit_number"].iloc[0]].sort_values("time_cycles").tail(30)

readings = one_engine.drop(columns=["unit_number", "RUL"], errors="ignore").to_dict(orient="records")
request_body = {
    "dataset_name": "FD004",
    "model_type": "gbr",
    "unit_number": int(one_engine["unit_number"].iloc[0]),
    "readings": readings,
}
print(json.dumps(request_body))