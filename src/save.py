from data_loader import load_test_data
from config import PROJECT_ROOT    
from train_baseline import run_baseline_training
from train_lstm import train_lstm_model
import joblib


if __name__ == "__main__":
    
    MODELS_DIR = PROJECT_ROOT / "models" / "FD001"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in ["FD001", "FD002", "FD003", "FD004"]:
        test_df = load_test_data(dataset_name)
        sample_units = test_df["unit_number"].unique()[:3]  # first 3 engines as examples
        sample_df = test_df[test_df["unit_number"].isin(sample_units)]
        sample_df.to_csv(PROJECT_ROOT / "models" / dataset_name / "sample_engines.csv", index=False)

    for dataset_name in ["FD001", "FD002", "FD003", "FD004"]:
        print(f"Saving {dataset_name}...")
        dataset_dir = PROJECT_ROOT / "models" / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # GBR
        gbr_model, _, _, kmeans, informative_sensors, feature_cols = run_baseline_training(dataset_name=dataset_name)
        joblib.dump(gbr_model, dataset_dir / "gbr_model.pkl")
        joblib.dump(kmeans, dataset_dir / "kmeans.pkl")
        joblib.dump(informative_sensors, dataset_dir / "informative_sensors.pkl")
        joblib.dump(feature_cols, dataset_dir / "feature_cols.pkl")

        # LSTM
        lstm_model, _, _, kmeans_l, informative_sensors_l, _, norm_stats = train_lstm_model(dataset_name=dataset_name)
        lstm_model.save(dataset_dir / "lstm_model.keras")
        joblib.dump(norm_stats, dataset_dir / "norm_stats.pkl")
        joblib.dump(kmeans_l, dataset_dir / "kmeans_lstm.pkl")
        joblib.dump(informative_sensors_l, dataset_dir / "informative_sensors_lstm.pkl")

    print("Done.")