import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------------------------------
st.set_page_config(page_title="System Failure Prediction", layout="centered")

# STRICT DATA CONTRACT (10 Features)
MODEL_SCHEMA = [
    "footfall", "tempMode", "AQ", "USS", "CS", 
    "VOC", "RP", "IP", "Temperature", "footfall_log"
]

MODEL_PATH = "pipeline_with_footfall_log.joblib"

# -------------------------------------------------------------------------------------------------
# MODEL LOADER (WITH SURGERY)
# -------------------------------------------------------------------------------------------------
@st.cache_resource
def load_system_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"CRITICAL: Artifact '{MODEL_PATH}' not found.")
        return None
    try:
        # 1. Load the full artifact
        pipeline = joblib.load(MODEL_PATH)
        
        # 2. PERFORM SURGERY: Extract the Inner Classifier
        # The outer pipeline is expanding data to 28 cols, but the inner one wants 10.
        # We bypass the outer 'preprocessor' and grab the 'classifier'.
        if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
            inner_model = pipeline.named_steps['classifier']
            return inner_model
            
        # Fallback: If surgery isn't possible, use the whole thing (unlikely to work)
        return pipeline
        
    except Exception as e:
        st.error(f"Model Integrity Error: {e}")
        return None

# -------------------------------------------------------------------------------------------------
# DATA GUARD
# -------------------------------------------------------------------------------------------------
def enforce_model_schema(df_raw):
    df_clean = df_raw.copy()

    # 1. Feature Engineering
    if "footfall" not in df_clean.columns:
        st.error("Missing 'footfall' column.")
        return None
    df_clean["footfall_log"] = np.log1p(df_clean["footfall"])

    # 2. Strict Filtering
    try:
        df_final = df_clean[MODEL_SCHEMA]
        return df_final
    except KeyError as e:
        missing = list(set(MODEL_SCHEMA) - set(df_clean.columns))
        st.error(f"Missing columns: {missing}")
        return None

# -------------------------------------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------------------------------------
def main():
    st.title("System Failure Prediction")
    st.markdown("### Operational Dashboard")
    
    # 1. Clear Cache Button (CRITICAL FOR THIS FIX)
    if st.sidebar.button("🧹 Force Reload Model"):
        st.cache_resource.clear()
        st.rerun()

    # 2. Load Model
    model = load_system_model()
    if model is None:
        st.stop()

    tabs = st.tabs(["Manual Diagnostics", "Batch Processing"])

    # --- MANUAL TAB ---
    with tabs[0]:
        with st.form("manual_form"):
            c1, c2 = st.columns(2)
            with c1:
                footfall = st.number_input("Footfall", 0, 10000, 100)
                temp_mode = st.slider("Temp Mode", 0, 10, 3)
                aq = st.slider("AQ", 0, 1000, 100)
                uss = st.slider("USS", 0, 1000, 100)
                cs = st.slider("CS", 0, 10, 5)
            with c2:
                voc = st.slider("VOC", 0, 1000, 100)
                rp = st.slider("RP", 0, 1000, 100)
                ip = st.slider("IP", 0, 10, 5)
                temp = st.number_input("Temperature", -50, 150, 25)
            
            submit = st.form_submit_button("Run Diagnostics", type="primary")

        if submit:
            raw_data = {
                "footfall": footfall, "tempMode": temp_mode, "AQ": aq,
                "USS": uss, "CS": cs, "VOC": voc,
                "RP": rp, "IP": ip, "Temperature": temp
            }
            df_ready = enforce_model_schema(pd.DataFrame([raw_data]))

            if df_ready is not None:
                try:
                    prediction = model.predict(df_ready)[0]
                    
                    st.divider()
                    if prediction == 1:
                        st.error("⚠️ FAILURE PREDICTED")
                        st.markdown("**Action:** Immediate maintenance required.")
                    else:
                        st.success("✅ SYSTEM STABLE")
                        st.markdown("**Action:** Continue monitoring.")
                except Exception as e:
                    st.error(f"Inference Failed: {e}")
                    st.warning("⚠️ Try clicking 'Force Reload Model' in the sidebar.")

    # --- BATCH TAB ---
    with tabs[1]:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file)
            df_ready = enforce_model_schema(df_upload)
            
            if df_ready is not None:
                try:
                    predictions = model.predict(df_ready)
                    
                    # Merge and Show
                    df_results = df_upload.copy()
                    df_results["Failure_Prediction"] = predictions
                    st.success("Batch Prediction Complete")
                    st.dataframe(df_results.head())
                    
                    csv = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Report", csv, "report.csv", "text/csv")
                except Exception as e:
                    st.error(f"Batch Failed: {e}")

if __name__ == "__main__":
    main()
