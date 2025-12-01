import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------------------------------------------------------
# ENTERPRISE CONFIGURATION
# -------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="System Failure Prediction",
    layout="centered",
    initial_sidebar_state="expanded"
)

# STRICT DATA CONTRACT
# The model will ONLY accept these columns in this EXACT order.
MODEL_SCHEMA = [
    "footfall",
    "tempMode",
    "AQ",
    "USS",
    "CS",
    "VOC",
    "RP",
    "IP",
    "Temperature",
    "footfall_log"
]

MODEL_PATH = "pipeline_with_footfall_log.joblib"

# -------------------------------------------------------------------------------------------------
# RESOURCE MANAGEMENT
# -------------------------------------------------------------------------------------------------
@st.cache_resource
def load_system_model():
    """
    Load the predictive pipeline with robust error handling.
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"CRITICAL: Artifact '{MODEL_PATH}' not found on server.")
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Model Integrity Error: {str(e)}")
        return None

# -------------------------------------------------------------------------------------------------
# DATA TRANSFORMATION PIPELINE
# -------------------------------------------------------------------------------------------------
def enforce_model_schema(df_raw):
    """
    Acts as a strict gatekeeper. 
    1. Validates prerequisites.
    2. Performs feature engineering (Log Transform).
    3. STRIPS extraneous columns (The 'Bouncer' Logic).
    """
    df_clean = df_raw.copy()

    # 1. Validation
    if "footfall" not in df_clean.columns:
        st.error("Schema Violation: Input missing 'footfall' column.")
        return None

    # 2. Engineering
    # Apply log transformation (log1p handles zeros safely)
    df_clean["footfall_log"] = np.log1p(df_clean["footfall"])

    # 3. Filtering (The Critical Fix)
    # We attempt to select ONLY the 10 required columns.
    # If any are missing, we catch the error immediately.
    try:
        df_final = df_clean[MODEL_SCHEMA]
        return df_final
    except KeyError as e:
        missing = list(set(MODEL_SCHEMA) - set(df_clean.columns))
        st.error(f"Schema Violation: Input is missing required features: {missing}")
        return None

# -------------------------------------------------------------------------------------------------
# USER INTERFACE & LOGIC
# -------------------------------------------------------------------------------------------------
def main():
    st.title("System Failure Prediction")
    st.markdown("### Predictive Maintenance Interface")
    
    # Load Model
    pipeline = load_system_model()
    if pipeline is None:
        st.stop()

    # Input Strategy
    tabs = st.tabs(["Manual Diagnostics", "Batch Processing"])

    # --- TAB 1: MANUAL ENTRY ---
    with tabs[0]:
        st.info("Configure sensor parameters below.")
        
        with st.form("diagnostic_form"):
            c1, c2 = st.columns(2)
            
            with c1:
                footfall = st.number_input("Footfall", 0, 10000, 100)
                temp_mode = st.slider("Temp Mode", 0, 10, 3)
                aq = st.slider("Air Quality (AQ)", 0, 1000, 100)
                uss = st.slider("Ultrasonic Sensor (USS)", 0, 1000, 100)
                cs = st.slider("Current Sensor (CS)", 0, 10, 5)

            with c2:
                voc = st.slider("VOC Level", 0, 1000, 100)
                rp = st.slider("RP", 0, 1000, 100)
                ip = st.slider("IP", 0, 10, 5)
                temp = st.number_input("Temperature (°C)", -50, 150, 25)

            # Hidden logic: We don't ask user for footfall_log, we calculate it.
            submit = st.form_submit_button("Run Diagnostics", type="primary")

        if submit:
            # Construct Dictionary
            raw_data = {
                "footfall": footfall, "tempMode": temp_mode, "AQ": aq,
                "USS": uss, "CS": cs, "VOC": voc,
                "RP": rp, "IP": ip, "Temperature": temp
            }
            
            # Process
            df_raw = pd.DataFrame([raw_data])
            df_ready = enforce_model_schema(df_raw)

            if df_ready is not None:
                try:
                    # Developer Mode: Show exactly what is being sent to the model
                    with st.expander("Developer Debug Data"):
                        st.write("Input Shape:", df_ready.shape)
                        st.write("Active Columns:", df_ready.columns.tolist())
                    
                    # Inference
                    prediction = pipeline.predict(df_ready)[0]
                    
                    st.divider()
                    if prediction == 1:
                        st.error("⚠️ ALERT: FAILURE PREDICTED")
                        st.markdown("**Recommendation:** Initiate emergency maintenance protocols.")
                    else:
                        st.success("✅ STATUS: SYSTEM STABLE")
                        st.markdown("**Recommendation:** Continue standard monitoring.")
                        
                except Exception as e:
                    st.error(f"Inference Engine Failed: {str(e)}")

    # --- TAB 2: BATCH UPLOAD ---
    with tabs[1]:
        st.info("Upload sensor logs for bulk analysis (CSV).")
        uploaded_file = st.file_uploader("Select CSV File", type=["csv"])

        if uploaded_file:
            try:
                df_upload = pd.read_csv(uploaded_file)
                
                # Process
                df_ready = enforce_model_schema(df_upload)
                
                if df_ready is not None:
                    # Inference
                    predictions = pipeline.predict(df_ready)
                    
                    # Result Merging
                    df_results = df_upload.copy()
                    df_results["Failure_Prediction"] = predictions
                    
                    # Visual Feedback
                    st.success("Processing Complete")
                    st.dataframe(df_results.head())
                    
                    # Export
                    csv = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Analysis Report",
                        csv,
                        "failure_report.csv",
                        "text/csv"
                    )
            except Exception as e:
                st.error(f"Batch Processing Error: {str(e)}")

if __name__ == "__main__":
    main()
