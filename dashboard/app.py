import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

API_URL = "http://127.0.0.1:8000/predict"
FEATURE_IMPORTANCE_URL = "http://127.0.0.1:8000/feature-importance"

st.title("Turbofan Engine RUL Prediction")


def rul_status(rul: float):
    if rul < 25:
        return "Urgent — schedule maintenance", "red"
    elif rul < 75:
        return "Monitor — approaching maintenance window", "orange"
    else:
        return "Healthy — no action needed", "green"


def status_badge(rul: float):
    label, color = rul_status(rul)
    st.markdown(
        f"<span style='background-color:{color}; color:white; padding:3px 10px; "
        f"border-radius:10px; font-size:0.85em;'>{label}</span>",
        unsafe_allow_html=True,
    )


if "history" not in st.session_state:
    st.session_state["history"] = []

dataset_name = st.selectbox("Dataset", ["FD001", "FD002", "FD003", "FD004"])

# --- Feature importance (GBR only) -------------------------------------------
with st.expander("GBR Feature Importance"):
    if st.button("Load feature importance"):
        resp = requests.get(f"{FEATURE_IMPORTANCE_URL}/{dataset_name}")
        if resp.status_code == 200:
            st.session_state["feature_importance"] = resp.json()["features"]
        else:
            st.error(f"Error: {resp.status_code} - {resp.text}")

    if "feature_importance" in st.session_state:
        top_n = st.slider("Show top N features", 5, 30, 15)
        fi = st.session_state["feature_importance"][:top_n]
        names = [f["feature"] for f in fi][::-1]
        scores = [f["importance"] for f in fi][::-1]

        fig_fi, ax_fi = plt.subplots(figsize=(8, max(3, top_n * 0.3)))
        ax_fi.barh(names, scores, color="teal")
        ax_fi.set_xlabel("Importance")
        st.pyplot(fig_fi)
        

st.divider()

# --- Sample engine selection -------------------------------------------------
try:
    sample_df = pd.read_csv(f"../models/{dataset_name}/sample_engines.csv")
    available_units = sorted(sample_df["unit_number"].unique())
except FileNotFoundError:
    sample_df = None
    available_units = []

if available_units:
    selected_unit = st.selectbox("Sample engine", available_units)
else:
    st.error(f"No sample_engines.csv found for {dataset_name}.")
    selected_unit = None

if st.button("Load sample engine") and selected_unit is not None:
    engine_data_full = (
        sample_df[sample_df["unit_number"] == selected_unit]
        .sort_values("time_cycles")
        .reset_index(drop=True)
    )
    st.session_state["engine_data_full"] = engine_data_full
    st.session_state["unit_id"] = int(selected_unit)
    st.session_state["loaded_dataset"] = dataset_name
    st.session_state.pop("gbr_result", None)
    st.session_state.pop("lstm_result", None)

# --- Main display --------------------------------------------------------
if "engine_data_full" not in st.session_state:
    st.info("Select an engine above and click 'Load sample engine' to get started.")
else:
    if st.session_state.get("loaded_dataset") != dataset_name:
        st.warning(
            f"Loaded sample is from {st.session_state['loaded_dataset']}, "
            f"but {dataset_name} is currently selected. Click 'Load sample engine' again "
            f"to load a matching sample, or switch the dataset back."
        )
    else:
        full_data = st.session_state["engine_data_full"]
        window_data = full_data.tail(30)

        st.write(
            f"Loaded engine {st.session_state['unit_id']} ({dataset_name}) — "
            f"{len(full_data)} total cycles"
        )
        st.dataframe(window_data)

        true_rul = None
        if "RUL" in full_data.columns and full_data["RUL"].notna().any():
            true_rul = float(full_data["RUL"].iloc[-1])
            st.metric("True RUL (ground truth)", f"{true_rul:.0f} cycles")

        # --- Multi-sensor selection for the chart ----------------------------
        sensor_cols = [c for c in full_data.columns if c.startswith("sensor")]
        default_sensors = [s for s in ["sensor_3", "sensor_9"] if s in sensor_cols] or sensor_cols[:1]
        selected_sensors = st.multiselect(
            "Sensors to plot (normalized 0-1 for comparison)",
            sensor_cols,
            default=default_sensors,
        )

        if selected_sensors:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

            for i, sensor in enumerate(selected_sensors):
                raw = full_data[sensor]
                span = raw.max() - raw.min()
                normalized = (raw - raw.min()) / span if span > 0 else raw * 0
                color = colors[i % len(colors)]

                ax.plot(full_data["time_cycles"], normalized, color=color, linewidth=1.3, label=sensor)
                ax.plot(
                    window_data["time_cycles"], normalized.tail(30),
                    color=color, linewidth=3, alpha=0.5,
                )

            last_cycle = full_data["time_cycles"].iloc[-1]
            marker_y = 1.08
            marker_colors = {"gbr_result": "royalblue", "lstm_result": "crimson"}
            for model_key, label in [("gbr_result", "GBR"), ("lstm_result", "LSTM")]:
                if model_key in st.session_state:
                    result = st.session_state[model_key]
                    color = marker_colors[model_key]
                    offset_x = -3 if model_key == "gbr_result" else 3
                    ax.scatter([last_cycle + offset_x], [marker_y], color=color, zorder=5, s=60, marker="v")
                    ax.annotate(
                        f"{label}: {result['predicted_RUL']:.1f}",
                        xy=(last_cycle + offset_x, marker_y),
                        xytext=(5, 8),
                        textcoords="offset points",
                        color=color,
                        fontsize=9,
                        fontweight="bold",
                    )

            ax.set_xlabel("Cycle")
            ax.set_ylabel("Normalized sensor value (0-1)")
            ax.set_ylim(-0.05, 1.2)
            ax.legend(loc="lower left", fontsize=8)
            st.pyplot(fig)
        else:
            st.info("Select at least one sensor above to see the chart.")

        # --- Predict with BOTH models at once --------------------------------
        if st.button("Predict RUL (GBR + LSTM)"):
            readings = window_data.drop(
                columns=["unit_number", "RUL"], errors="ignore"
            ).to_dict(orient="records")

            for model_type, key in [("gbr", "gbr_result"), ("lstm", "lstm_result")]:
                request_body = {
                    "dataset_name": dataset_name,
                    "model_type": model_type,
                    "unit_number": st.session_state["unit_id"],
                    "readings": readings,
                }
                response = requests.post(API_URL, json=request_body)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state[key] = result

                    st.session_state["history"].append({
                        "dataset": dataset_name,
                        "engine": st.session_state["unit_id"],
                        "model": model_type.upper(),
                        "predicted_RUL": result["predicted_RUL"],
                        "true_RUL": true_rul,
                        "error": abs(result["predicted_RUL"] - true_rul) if true_rul is not None else None,
                        "capped": result.get("capped", False),
                    })
                else:
                    st.error(f"{model_type.upper()} error: {response.status_code} - {response.text}")
                    st.session_state.pop(key, None)

            st.rerun()

        # --- Side-by-side results, with status badges -------------------------
        if "gbr_result" in st.session_state or "lstm_result" in st.session_state:
            st.subheader("Results")

            col_gbr, col_lstm = st.columns(2)

            with col_gbr:
                st.markdown("**Gradient Boosting Regressor**")
                if "gbr_result" in st.session_state:
                    r = st.session_state["gbr_result"]
                    st.metric("Predicted RUL", f"{r['predicted_RUL']:.2f}")
                    status_badge(r["predicted_RUL"])
                    if true_rul is not None:
                        st.caption(f"Error: {abs(r['predicted_RUL'] - true_rul):.2f} cycles")
                    if r.get("capped", False):
                        st.warning("At/near training cap (125) — unreliable beyond this point.")
                else:
                    st.write("—")

            with col_lstm:
                st.markdown("**LSTM**")
                if "lstm_result" in st.session_state:
                    r = st.session_state["lstm_result"]
                    st.metric("Predicted RUL", f"{r['predicted_RUL']:.2f}")
                    status_badge(r["predicted_RUL"])
                    if true_rul is not None:
                        st.caption(f"Error: {abs(r['predicted_RUL'] - true_rul):.2f} cycles")
                    if r.get("capped", False):
                        st.warning("At/near training cap (125) — unreliable beyond this point.")
                else:
                    st.write("—")

            if "gbr_result" in st.session_state and "lstm_result" in st.session_state and true_rul is not None:
                gbr_err = abs(st.session_state["gbr_result"]["predicted_RUL"] - true_rul)
                lstm_err = abs(st.session_state["lstm_result"]["predicted_RUL"] - true_rul)
                better = "LSTM" if lstm_err < gbr_err else "GBR"
                st.info(f"**{better}** was closer to the true RUL on this engine (by {abs(gbr_err - lstm_err):.2f} cycles).")

# --- Prediction history, across engines/datasets within this session --------
st.divider()
st.subheader("Prediction History (this session)")

if not st.session_state["history"]:
    st.caption("No predictions made yet. Load an engine and click 'Predict RUL' to start building history.")
else:
    history_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(history_df, use_container_width=True)

    if st.button("Clear history"):
        st.session_state["history"] = []
        st.rerun()

    has_true_rul = history_df["true_RUL"].notna().any()
    if has_true_rul:
        st.subheader("Predicted vs. True RUL — across all engines predicted this session")

        plot_df = history_df.dropna(subset=["true_RUL"])
        fig2, ax2 = plt.subplots(figsize=(6, 6))

        for model, color in [("GBR", "royalblue"), ("LSTM", "crimson")]:
            subset = plot_df[plot_df["model"] == model]
            if len(subset):
                ax2.scatter(subset["true_RUL"], subset["predicted_RUL"], color=color, label=model, alpha=0.7)

        lims = [0, max(plot_df["true_RUL"].max(), plot_df["predicted_RUL"].max()) + 10]
        ax2.plot(lims, lims, linestyle="--", color="gray", label="Perfect prediction")
        ax2.set_xlabel("True RUL")
        ax2.set_ylabel("Predicted RUL")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.caption("Load engines with known true RUL (from the real test set) to see this comparison.")