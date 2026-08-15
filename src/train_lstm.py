from keras import layers
from keras.models import Sequential
from sequence_builder import build_train_sequences, build_test_sequences, simulate_test_sequences
from feature_engineering import select_informative_sensors
from data_loader import load_train_data
from preprocessing import compute_rul, fit_regime_clusters, assign_regimes, fit_regime_norm_stats, normalize_by_regime
from train_baseline import split_by_unit
from keras.callbacks import EarlyStopping
from scoring import evaluate, phm08_per_engine



def build_lstm_model(window_length, n_features, dropout_rate=0.2, lstm_units_1=64, lstm_units_2=32):
    # TODO: first LSTM layer, 64 units, return_sequences=True (so the next LSTM layer gets a full sequence, not just one vector)
    # TODO: dropout layer
    # TODO: second LSTM layer, 32 units, return_sequences=False (default - collapses to one vector, since this is the last recurrent layer)
    # TODO: dropout layer
    # TODO: dense output layer - 1 unit, no activation (this is regression, not classification)
    
    model = Sequential([
        layers.Input(shape=(window_length, n_features)),
        layers.LSTM(lstm_units_1, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(lstm_units_2, return_sequences=False),
        layers.Dropout(dropout_rate),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def evaluate_lstm(model, val_units_df, informative_sensors, window_length=30, random_seed=42):
    X_val_sim, y_val_sim, val_unit_ids = simulate_test_sequences(
        val_units_df, informative_sensors, window_length, random_seed=random_seed
    )
    preds = model.predict(X_val_sim).flatten()

    metrics = evaluate(y_val_sim, preds)
    per_engine_df = phm08_per_engine(y_val_sim, preds, val_unit_ids)

    return metrics, per_engine_df

def train_lstm_model(dataset_name="FD001", window_length=30, test_size=0.2, random_seed=42):
    # Load and preprocess the data
    train_df = load_train_data(dataset_name)
    train_df = compute_rul(train_df)

    kmeans = fit_regime_clusters(train_df, n_regimes=1)
    train_df = assign_regimes(train_df, kmeans)

    all_sensor_columns = [f"sensor_{i}" for i in range(1, 22)]
    informative_sensors, dropped_sensors = select_informative_sensors(train_df, all_sensor_columns)

    # LSTM needs normalized inputs - unlike the GBR baseline, which never normalized
    norm_stats = fit_regime_norm_stats(train_df, all_sensor_columns)
    train_df = normalize_by_regime(train_df, norm_stats, all_sensor_columns)

    # Split by unit BEFORE building sequences - no engine's cycles should leak across train/val
    train_units_df, val_units_df = split_by_unit(train_df, test_size=test_size, random_seed=random_seed)

    X_train, y_train = build_train_sequences(train_units_df, informative_sensors, window_length)
    X_val, y_val = build_train_sequences(val_units_df, informative_sensors, window_length)

    model_v2 = build_lstm_model(window_length=30, n_features=len(informative_sensors), lstm_units_1=64, lstm_units_2=32, dropout_rate=0.3)
    model = build_lstm_model(window_length=window_length, n_features=len(informative_sensors))

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=[early_stop],
        batch_size=32,
        verbose="1"
    )

    metrics, per_engine_df = evaluate_lstm(model, val_units_df, informative_sensors)

    return model, per_engine_df, metrics, kmeans, informative_sensors, history, norm_stats

if __name__ == "__main__":
    model = train_lstm_model()
    