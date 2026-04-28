# ACCESS ZIP FILES HERE : 
https://drive.google.com/drive/folders/1fm0BF7NthGBoaQ2qrlfeu7sARIdL1NMK?usp=sharing


# DEPHIDES: Deep Learning Based Phishing Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the implementation of the **DEPHIDES** (Deep Hierarchical Phishing Detection System) framework, a robust deep-learning-based approach for identifying malicious URLs with over **97% accuracy**.

##  Key Features

- **Character-Level Embeddings**: Processes URLs as raw character sequences to capture fine-grained patterns and handle zero-day obfuscations.
- **Dual-Stream Analysis**: Combines deep character embeddings with a specialized taxonomy of **30 hand-crafted lexical features**.
- **Multi-Model Comparison**: Evaluates 10 distinct architectures across 5 neural network paradigms:
    - **CNN** (Convolutional Neural Networks) - 17rd-layer Complex variant.
    - **RNN/BRNN** (Bidirectional Long Short-Term Memory) - 7-layer stacked variants.
    - **Attention** - Self-attention mechanisms to focus on suspicious URL segments.
    - **ANN** (Artificial Neural Networks) - Deep baseline dense networks.
- **Traditional ML Support**: Includes implementations for Random Forest, SVM, and Logistic Regression.

## Repository Structure

- `models/`: Architecture definitions for all neural network types.
    - `Traditional_ML/`: Classical machine learning models and predictors.
- `scripts/`: specialized training scripts for individual architectures (e.g., `train_cnn.py`).
- `dataset/`: Training and testing data (character indices and raw labels).
- `evaluate.py`: The unified entry point for training and evaluating DL models.
- `CNN_RNN_Training.ipynb`: Jupyter notebook for interactive model training and visualization.
- `feature_extraction.py`: The core logic for extracting the 30 hand-crafted features and character sequences.
- `DEPHIDES_Report_Extended.tex`: A comprehensive **4-page technical report** detailing the methodology and results.

## Usage Instructions

### 1. Installation
Install the necessary dependencies using:
```bash
pip install -r requirements.txt
```

### 2. Training and Evaluation
Use `evaluate.py` to train and test a model. You can specify the model type, embedding dimensions, and sequence length.
```bash
# Train the 17-layer CNN Complex model
python evaluate.py --model cnn_complex --sequence_length 200 --epochs 10

# Train the 7-layer BRNN Complex model
python evaluate.py --model brnn_complex --sequence_length 200 --epochs 5
```

### 3. Feature Extraction Details (The 30 Features)
Our system utilizes 30 specific lexical features including:
- **Structural**: Length metrics, path depth, subdomain counts.
- **Statistical**: Shannon Entropy of URL/Domain, character ratios (Letter/Digit/Special).
- **Security Protocols**: IP address usage, non-standard port detection, HTTPS/HTTP status.
- **Obfuscation Markers**: URL shortener detection and character repetition counts.


## Results Summary
The system achieves peak performance using the **CNN Complex** and **BRNN Complex** models, reaching an accuracy of **97.4%**. Detailed metrics and GPU utilization plots are generated automatically in the `test_results/` directory after each run.

---
*Developed for research into character-level phishing detection and deep sequence modeling.*