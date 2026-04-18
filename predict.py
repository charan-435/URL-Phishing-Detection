import argparse
import os
import sys
import tensorflow as tf
import numpy as np
from feature_extraction import FeatureExtractor

# whitelist for trusted domains
TRUSTED = ["google.com", "facebook.com", "apple.com", "microsoft.com", "amazon.com", "github.com"]

# analyze a single url and print results
def check_url(url, model_path, char_index_path, seq_len=512):
    print(f"--- Checking: {url} ---")
    
    # clean url
    u = url.lower().replace("http://", "").replace("https://", "").replace("www.", "").strip("/")
    
    # check whitelist
    if u in TRUSTED or any(u.startswith(t + "/") for t in TRUSTED):
        print("\n" + "="*30)
        print("  RESULT:     LEGITIMATE")
        print("  CONFIDENCE: 100.00% (Trusted)")
        print("="*30)
        return

    # paths check
    if not os.path.exists(char_index_path):
        print("err: char_index missing. train first!")
        return

    # load extractor
    extractor = FeatureExtractor(char_index_path=char_index_path)
    
    # clean url
    clean_url = url.lower().replace("http://", "").replace("https://", "")
    extractor.urls = [clean_url]
    
    # model check
    if not os.path.exists(model_path):
        print("err: model file missing!")
        return
        
    print("loading model...")
    model = tf.keras.models.load_model(model_path)
    
    # extract and predict
    features = extractor.get_sequences(sequence_length=seq_len)
    
    print("predicting...")
    prediction = model.predict(features, verbose=0)[0][0]
    
    # results check
    label = "PHISHING" if prediction < 0.5 else "LEGITIMATE"
    confidence = (1 - prediction) if prediction < 0.5 else prediction
    
    print("\n" + "="*30)
    print(f"  RESULT:     {label}")
    print(f"  CONFIDENCE: {confidence*100:.2f}%")
    print("="*30)
    
    if label == "PHISHING":
        print("be careful, looks fake!")
    else:
        print("looks ok.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--model", type=str, default="test_results/baseline/cnn_complex/model_all.keras")
    parser.add_argument("--char_index", type=str, default="dataset/char_index")
    args = parser.parse_args()

    check_url(args.url, args.model, args.char_index)
