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

# -------------------------------------------------------------------------------------------------
# FEATURE ENGINEERING LOGIC (The Fix)
# -------------------------------------------------------------------------------------------------
def preprocess_input(df):
    """
    Transforms raw features into the features expected by the model.
    specifically: calculates 'footfall_log'.
    """
    df_processed = df.copy()
    
    # Check if 'footfall' exists to avoid errors
    if 'footfall' in df_processed.columns:
        # We use log1p (log(x+1)) to handle cases where footfall is 0 safely
        df_processed['footfall_log'] = np.log1p(df_processed['footfall'])
    else:
        # If user uploads CSV with 'footfall_log' already, we are fine.
        # If neither exists, we have a problem.
        if 'footfall_log' not in df_processed.columns:
            st.error("Input must contain 'footfall' column.")
            return None

    return df_processed

# -------------------------------------------------------------------------------------------------
# UI & INPUTS
# -------------------------------------------------------------------------------------------------
st.title("⚙️ System Failure Predictor")
st.markdown("Early warning system for predictive maintenance.")

if pipeline is None:
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["🎛️ Manual Input", "📂 Batch Upload"])

with tab1:
    st.subheader("Sensor Readings")
    col1, col2 = st.columns(2)
    
    single_input = {}
    
    with col1:
        # We capture the RAW footfall here
        single_input['footfall'] = st.number_input("Footfall (0-10000)", 0, 10000, 100)
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
        # 1. Create DataFrame
        input_df = pd.DataFrame([single_input])
        
        # 2. Apply Engineering (Create footfall_log)
        processed_df = preprocess_input(input_df)
        
        if processed_df is not None:
            try:
                # 3. Predict
                prediction = pipeline.predict(processed_df)[0]
                
                st.markdown("---")
                if prediction == 1:
                    st.error("### ⚠️ PREDICTION: FAILURE IMMINENT")
                    st.warning("Action required immediately.")
                else:
                    st.success("### ✅ PREDICTION: SYSTEM STABLE")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Technical Detail: Ensure the model pipeline expects these exact columns.")

with tab2:
    st.subheader("Batch Processing")
    uploaded_file = st.file_uploader("Upload Sensor CSV", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Preprocess the batch
            process_df = preprocess_input(df)
            
            if process_df is not None:
                predictions = pipeline.predict(process_df)
                
                df['Failure_Prediction'] = predictions
                st.write("Results Preview:")
                st.dataframe(df.head())
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Full Results", csv, "failure_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")
