# app.py - simple flask server for phishing detection
import os
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from feature_extraction import FeatureExtractor

app = Flask(__name__)

# config
MODEL_PATH = "test_results/baseline/cnn_complex/model_all.keras"
INDEX_PATH = "dataset/char_index"
SEQ_LEN = 512

# top domains to avoid false positives
WHITELIST = ["google.com", "facebook.com", "apple.com", "microsoft.com", "amazon.com", "github.com"]

# load model once at startup
print("loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
extractor = FeatureExtractor(char_index_path=INDEX_PATH)

@app.route("/")
def home():
    # main page
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check():
    # api endpoint to check a url
    url = request.json.get("url", "")
    if not url:
        return jsonify({"error": "no url provided"}), 400
    
    # clean and extract
    url_clean = url.lower().replace("http://", "").replace("https://", "").replace("www.", "").strip("/")
    
    # check whitelist first
    for trusted in WHITELIST:
        if url_clean == trusted or url_clean.startswith(trusted + "/"):
            return jsonify({
                "url": url,
                "label": "LEGITIMATE",
                "confidence": 100.0,
                "note": "trusted domain"
            })

    extractor.urls = [url_clean]
    features = extractor.get_sequences(sequence_length=SEQ_LEN)
    
    # predict
    pred = model.predict(features, verbose=0)[0][0]
    
    # result logic
    is_phishing = pred < 0.5
    conf = (1 - pred) if is_phishing else pred
    
    return jsonify({
        "url": url,
        "label": "PHISHING" if is_phishing else "LEGITIMATE",
        "confidence": round(float(conf) * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
