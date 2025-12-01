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

# The EXACT 10 columns the model requires
# (9 Raw Features + 1 Engineered Feature)
FINAL_FEATURES = [
    'footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature', 
    'footfall_log'
]

# -------------------------------------------------------------------------------------------------
# CACHED RESOURCE LOADING
# -------------------------------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_FILE):
        st.error(f"Critical Error: Model file '{MODEL_FILE}' not found.")
        return None
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipeline = load_pipeline()

# -------------------------------------------------------------------------------------------------
# FEATURE ENGINEERING & FILTERING (The Fix)
# -------------------------------------------------------------------------------------------------
def preprocess_input(df):
    """
    1. Calculates 'footfall_log'.
    2. FILTERS the dataframe to keep ONLY the 10 required columns.
    """
    df_processed = df.copy()
    
    # 1. Feature Engineering (Calculate the log)
    if 'footfall' in df_processed.columns:
        # Use log1p to handle zeros safely
        df_processed['footfall_log'] = np.log1p(df_processed['footfall'])
    else:
        # If the input doesn't have footfall, we can't proceed
        st.error("Input data missing required column: 'footfall'")
        return None

    # 2. Strict Column Filtering (The "Bouncer")
    # This removes the extra 18 columns causing the crash
    try:
        df_final = df_processed[FINAL_FEATURES]
        return df_final
    except KeyError as e:
        # This catches if one of the REQUIRED columns is missing
        missing = list(set(FINAL_FEATURES) - set(df_processed.columns))
        st.error(f"Missing required columns for prediction: {missing}")
        return None

# -------------------------------------------------------------------------------------------------
# UI & INPUTS
# -------------------------------------------------------------------------------------------------
st.title("System Failure Predictor")
st.markdown("Early warning system for predictive maintenance.")

if pipeline is None:
    st.stop()

tab1, tab2 = st.tabs(["Manual Input", "Batch Upload"])

with tab1:
    st.subheader("Sensor Readings")
    col1, col2 = st.columns(2)
    
    single_input = {}
    
    with col1:
        single_input['footfall'] = st.number_input("Footfall (0-10000)", 0, 10000, 100)
        single_input['tempMode'] = st.slider("Temp Mode (0-10)", 0, 10, 3)
        single_input['AQ'] = st.slider("Air Quality (AQ)", 0, 1000, 100)
        single_input['USS'] = st.slider("Ultrasonic Sensor (USS)", 0, 1000, 100)
        single_input['CS'] = st.slider("Current Sensor (CS)", 0, 10, 5)
        
    with col2:
        single_input['VOC'] = st.slider("VOC Level", 0, 1000, 100)
        single_input['RP'] = st.slider("RP", 0, 1000, 100)
        single_input['IP'] = st.slider("IP", 0, 10, 5)
        single_input['Temperature'] = st.number_input("Temperature", -50, 150, 25)

    if st.button("Analyze Sensors", type="primary"):
        input_df = pd.DataFrame([single_input])
        
        # Preprocess
        processed_df = preprocess_input(input_df)
        
        if processed_df is not None:
            try:
                prediction = pipeline.predict(processed_df)[0]
                
                st.markdown("---")
                if prediction == 1:
                    st.error("PREDICTION: FAILURE IMMINENT")
                    st.warning("Action required immediately.")
                else:
                    st.success("PREDICTION: SYSTEM STABLE")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tab2:
    st.subheader("Batch Processing")
    uploaded_file = st.file_uploader("Upload Sensor CSV", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Preprocess
            process_df = preprocess_input(df)
            
            if process_df is not None:
                predictions = pipeline.predict(process_df)
                
                # Attach predictions to original for context
                df['Failure_Prediction'] = predictions
                
                # Reorder columns to show prediction first
                cols = ['Failure_Prediction'] + [c for c in df.columns if c != 'Failure_Prediction']
                df = df[cols]
                
                st.success("Analysis Complete")
                st.dataframe(df.head())
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Full Results", csv, "failure_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")
