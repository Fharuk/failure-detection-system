import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Predictive Failure Detection", layout="centered")
MODEL_FILE = "pipeline_with_footfall_log.joblib"

# -------------------------------------------------------------------------------------------------
# CACHED RESOURCE LOADING
# -------------------------------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    """
    Load the pipeline once and keep it in memory.
    """
    if not os.path.exists(MODEL_FILE):
        st.error(f"🚨 Critical Error: Model file '{MODEL_FILE}' not found.")
        return None
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipeline = load_pipeline()

# Features expected by the pipeline
pipeline_features = ['footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature']

# -------------------------------------------------------------------------------------------------
# UI & INPUTS
# -------------------------------------------------------------------------------------------------
st.title("⚙️ System Failure Predictor")
st.markdown("Early warning system for predictive maintenance.")

if pipeline is None:
    st.stop()

# Tabs for different input methods
tab1, tab2 = st.tabs(["🎛️ Manual Input", "📂 Batch Upload"])

with tab1:
    st.subheader("Sensor Readings")
    col1, col2 = st.columns(2)
    
    single_input = {}
    
    # We split inputs into two columns for a cleaner look
    with col1:
        single_input['footfall'] = st.number_input("Footfall (0-1000)", 0, 10000, 100)
        single_input['tempMode'] = st.slider("Temp Mode (0-10)", 0, 10, 3)
        single_input['AQ'] = st.slider("Air Quality (AQ)", 0, 1000, 100)
        single_input['USS'] = st.slider("Ultrasonic Sensor (USS)", 0, 1000, 100)
        
    with col2:
        single_input['CS'] = st.slider("Current Sensor (CS)", 0, 10, 5)
        single_input['VOC'] = st.slider("VOC Level", 0, 1000, 100)
        single_input['RP'] = st.slider("RP", 0, 1000, 100)
        single_input['IP'] = st.slider("IP", 0, 10, 5)
        single_input['Temperature'] = st.number_input("Temperature", -50, 150, 25)

    if st.button("Analyze Sensors", type="primary"):
        input_df = pd.DataFrame([single_input])
        # Ensure correct column order
        input_df = input_df[pipeline_features]
        
        try:
            prediction = pipeline.predict(input_df)[0]
            st.markdown("---")
            if prediction == 1:
                st.error("### ⚠️ PREDICTION: FAILURE IMMINENT")
                st.warning("Action required immediately.")
            else:
                st.success("### ✅ PREDICTION: SYSTEM STABLE")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
    st.subheader("Batch Processing")
    uploaded_file = st.file_uploader("Upload Sensor CSV", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            missing_cols = set(pipeline_features) - set(df.columns)
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                # Keep only relevant columns
                process_df = df[pipeline_features].copy()
                predictions = pipeline.predict(process_df)
                
                df['Failure_Prediction'] = predictions
                st.write("Results Preview:")
                st.dataframe(df.head())
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Full Results", csv, "failure_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")