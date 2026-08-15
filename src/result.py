import pandas as pd
from os import path
from config import RAW_DATA_DIR

results = [
    {"dataset": "FD001", "model": "GBR", "RMSE": 19.479433, "PHM08_score": 1527.985198},
    {"dataset": "FD001", "model": "LSTM", "RMSE": 15.634039, "PHM08_score": 428.521843},
    {"dataset": "FD003", "model": "GBR", "RMSE": 21.143442, "PHM08_score": 2096.326473},
    {"dataset": "FD003", "model": "LSTM", "RMSE": 14.145102, "PHM08_score": 330.107386},
]
results_df = pd.DataFrame(results)
results_df.to_csv("results\metrics_summary.csv", index=False)
print(results_df)