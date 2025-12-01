# make_reqs.py
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("streamlit\npandas\nnumpy\njoblib\nscikit-learn")
print("requirements.txt created successfully.")