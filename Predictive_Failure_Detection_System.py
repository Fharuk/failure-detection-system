import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Application Configuration
st.set_page_config(
    page_title="System Failure Prediction",
    layout="centered"
)

# Constants
MODEL_FILENAME = "pipeline_with_footfall_log.joblib"
REQUIRED_MODEL_FEATURES = [
    "footfall", "tempMode", "AQ", "USS", "CS", 
    "VOC", "RP", "IP", "Temperature", "footfall_log"
]

@st.cache_resource
def load_prediction_pipeline():
    """
    Load the serialized machine learning pipeline from disk.
    Uses caching to prevent reloading on every interaction.
    """
    if not os.path.exists(MODEL_FILENAME):
        st.error(f"Critical Error: Model artifact '{MODEL_FILENAME}' not found.")
        return None
    try:
        pipeline = joblib.load(MODEL_FILENAME)
        return pipeline
    except Exception as e:
        st.error(f"Failed to load model architecture: {str(e)}")
        return None

def transform_input_data(df_input):
    """
    Apply feature engineering and enforce strict schema compliance.
    
    1. Validates presence of raw 'footfall' column.
    2. Calculates 'footfall_log' using log1p transformation.
    3. Filters dataset to strictly match the 10 features expected by the model.
    """
    df_transformed = df_input.copy()

    # Validate raw dependency
    if "footfall" not in df_transformed.columns:
        st.error("Validation Error: Input data is missing the 'footfall' column.")
        return None

    # Feature Engineering
    # Using log1p to handle potential zero values in footfall safely
    df_transformed["footfall_log"] = np.log1p(df_transformed["footfall"])

    # Schema Enforcement
    # Filter only the columns the model was trained on. 
    # This prevents dimension mismatch errors if the input has extra columns.
    try:
        df_final = df_transformed[REQUIRED_MODEL_FEATURES]
        return df_final
    except KeyError as e:
        missing_features = list(set(REQUIRED_MODEL_FEATURES) - set(df_transformed.columns))
        st.error(f"Schema Error: Missing required features: {missing_features}")
        return None

def main():
    st.title("System Failure Prediction")
    st.markdown("Predictive maintenance interface for sensor data analysis.")

    # Load Model
    pipeline = load_prediction_pipeline()
    if pipeline is None:
        st.stop()

    # Input Method Selection
    tab_manual, tab_batch = st.tabs(["Manual Entry", "Batch Upload"])

    # Manual Entry Tab
    with tab_manual:
        st.subheader("Sensor Parameters")
        
        with st.form("manual_input_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                footfall = st.number_input("Footfall", min_value=0, max_value=10000, value=100)
                temp_mode = st.slider("Temp Mode", 0, 10, 3)
                aq = st.slider("Air Quality (AQ)", 0, 1000, 100)
                uss = st.slider("Ultrasonic Sensor (USS)", 0, 1000, 100)
                cs = st.slider("Current Sensor (CS)", 0, 10, 5)
            
            with col2:
                voc = st.slider("VOC Level", 0, 1000, 100)
                rp = st.slider("RP", 0, 1000, 100)
                ip = st.slider("IP", 0, 10, 5)
                temperature = st.number_input("Temperature (C)", min_value=-50, max_value=150, value=25)

            submit_btn = st.form_submit_button("Run Prediction", type="primary")

        if submit_btn:
            # Construct DataFrame from manual inputs
            # Note: We do not add footfall_log here; the transformer handles it.
            input_data = {
                "footfall": footfall, "tempMode": temp_mode, "AQ": aq,
                "USS": uss, "CS": cs, "VOC": voc,
                "RP": rp, "IP": ip, "Temperature": temperature
            }
            
            df_manual = pd.DataFrame([input_data])
            df_processed = transform_input_data(df_manual)

            if df_processed is not None:
                try:
                    prediction = pipeline.predict(df_processed)[0]
                    
                    st.divider()
                    st.subheader("Analysis Result")
                    if prediction == 1:
                        st.error("Status: FAILURE IMMINENT")
                        st.markdown("**Action Required:** Schedule immediate maintenance.")
                    else:
                        st.success("Status: SYSTEM STABLE")
                        st.markdown("**Action Required:** None. System operating within normal parameters.")
                except Exception as e:
                    st.error(f"Prediction execution failed: {str(e)}")

    # Batch Upload Tab
    with tab_batch:
        st.subheader("Bulk File Processing")
        uploaded_file = st.file_uploader("Upload sensor log (CSV)", type=["csv"])

        if uploaded_file:
            try:
                df_raw = pd.read_csv(uploaded_file)
                df_processed = transform_input_data(df_raw)

                if df_processed is not None:
                    # Run Inference
                    predictions = pipeline.predict(df_processed)
                    
                    # Merge results with original data for context
                    df_results = df_raw.copy()
                    df_results["Failure_Prediction"] = predictions
                    
                    # Move prediction column to the front
                    cols = ["Failure_Prediction"] + [c for c in df_results.columns if c != "Failure_Prediction"]
                    df_results = df_results[cols]
                    
                    st.success("Batch processing complete.")
                    st.dataframe(df_results.head())
                    
                    # Export
                    csv_data = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Analysis Report",
                        data=csv_data,
                        file_name="failure_predictions.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"File processing error: {str(e)}")

if __name__ == "__main__":
    main()
