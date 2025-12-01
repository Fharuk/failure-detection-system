import joblib
import pandas as pd
import numpy as np
import os

MODEL_FILE = "pipeline_with_footfall_log.joblib"
print(f"Testing Model: {MODEL_FILE}")

# 1. Load Model
if not os.path.exists(MODEL_FILE):
    print("❌ Model file not found!")
    exit()

pipeline = joblib.load(MODEL_FILE)
print("✅ Model Loaded.")

# 2. Create EXACT 10-column Dummy Data
cols = [
    "footfall", "tempMode", "AQ", "USS", "CS", 
    "VOC", "RP", "IP", "Temperature", "footfall_log"
]
# Create a row of zeros
df_dummy = pd.DataFrame([np.zeros(10)], columns=cols)

print(f"Input Shape: {df_dummy.shape}")
print(f"Columns: {df_dummy.columns.tolist()}")

# 3. Force Prediction
try:
    pred = pipeline.predict(df_dummy)
    print("\n🎉 SUCCESS! The model accepted the 10 columns.")
    print(f"Prediction: {pred[0]}")
except Exception as e:
    print("\n❌ FAILURE! The model rejected the 10 columns.")
    print(f"Error Message: {e}")
```

**Run it:**
```bash
python test_prediction.py
```

* **If this FAILS:** Your `joblib` file is corrupted or is not what we think it is.
* **If this SUCCEEDS:** Your Streamlit app is running an **old version of the code** (cached).

### Step 2: Fix the App (Cache Clearing)

If the test above succeeded, your Streamlit app is "stuck" using an old model object from memory. We need to force it to forget everything.

Add this **Cache Clearing Button** to the top of your `app.py` (inside `main()`):

```python
    # ... inside main() ...
    st.title("System Failure Prediction")
    
    # ADD THIS BUTTON TEMPORARILY
    if st.button("🧹 FORCE RESET CACHE"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
```

### Step 3: Verify You Are Editing the Right File

Change the title in `app.py` to something obvious:

```python
st.title("System Failure Prediction - DEBUG MODE ACTIVE")
