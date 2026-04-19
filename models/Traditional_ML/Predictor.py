import joblib
import numpy as np
from urllib.parse import urlparse
import re
from feature_extraction import FeatureExtractor

# =========================
# SAME FEATURE FUNCTION (VERY IMPORTANT)
# =========================
extractor = FeatureExtractor()
def extract_features(url):
    features = []

    features = extractor._extract_one(url)

    

    return features


# =========================
# LOAD SCALER
# =========================

scaler = joblib.load(r"output_data\feature_scaler.joblib")

# =========================
# LOAD MODELS
# =========================

models = {
    "RF": joblib.load(r"output_data\RF\RF_model.joblib"),
    "SVM": joblib.load(r"output_data\SVM\SVM_model.joblib"),
    "LR": joblib.load(r"output_data\LR\LR_model.joblib"),
    "NB": joblib.load(r"output_data\NB\NB_model.joblib")
}

# =========================
# INPUT
# =========================

url = input("Enter URL: ").strip().lower()
url = url.replace("http://", "").replace("https://", "")

# =========================
# PROCESS
# =========================

features = extract_features(url)
features = np.array(features).reshape(1, -1)

# scale (important)
features = scaler.transform(features)

# =========================
# PREDICT
# =========================

print("\n🔍 Predictions:\n")

for name, model in models.items():
    pred = model.predict(features)[0]

    label = "Phishing ⚠️" if pred == 0 else "Legitimate ✅"

    print(f"{name}: {label}")