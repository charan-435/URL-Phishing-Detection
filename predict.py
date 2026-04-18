import argparse
import os
import sys
import tensorflow as tf
import numpy as np
from feature_extraction import FeatureExtractor

# analyze a single url and print results
def check_url(url, model_path, char_index_path, seq_len=512):
    print(f"--- Checking: {url} ---")
    
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
