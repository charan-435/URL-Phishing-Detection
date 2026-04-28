# DEPHIDES: Deep Learning Based Phishing Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the implementation of the **DEPHIDES** (Deep Hierarchical Phishing Detection System) framework, as proposed in the research paper **"DEPHIDES: Deep Learning Based Phishing Detection System"** published in *IEEE Access* (2024). It provides a robust deep-learning-based approach for identifying malicious URLs with over **97% accuracy**.

##  Key Features

- **Character-Level Embeddings**: Processes URLs as raw character sequences to capture fine-grained patterns and handle zero-day obfuscations.
- **Lexical Feature Support**: Includes a specialized taxonomy of **30 hand-crafted lexical features** for traditional ML models and future multi-stream extensions.
- **Multi-Model Comparison**: Evaluates 10 distinct architectures across 5 neural network paradigms:
    - **CNN** (Convolutional Neural Networks) - 17-layer Complex variant.
    - **RNN/BRNN** (Bidirectional Long Short-Term Memory) - 7-layer stacked variants.
    - **Attention** - Self-attention mechanisms to focus on suspicious URL segments.
    - **ANN** (Artificial Neural Networks) - Deep baseline dense networks.
- **Traditional ML Support**: Includes implementations for Random Forest, SVM, Logistic Regression, and Naive Bayes.

## Repository Structure

- `models/`: Architecture definitions for all neural network types.
    - `cnn/`, `rnn/`, `brnn/`, `att/`, `ann/`: Deep learning model definitions (`base` and `complex` variants).
    - `Traditional_ML/`: Classical machine learning models and predictors.
- `scripts/`: Specialized training scripts for individual architectures (e.g., `train_cnn.py`).
- `notebooks/`: Jupyter notebooks for interactive training, EDA, and Kaggle-based experiments.
- `dataset/`: Training and testing data (character indices and raw labels).
    - `train/`, `val/`, `test/`: Partitioned dataset files.
    - `char_index`: Mapping for character-level tokenization.
- `test_results/`: Automatically generated metrics, confusion matrices, and GPU utilization plots.
- `evaluate.py`: The unified entry point for training and evaluating Deep Learning models.
- `feature_extraction.py`: The core logic for extracting the 30 hand-crafted features and character sequences.
- `Presentation.pdf`: Technical presentation/paper detailing the research.
- `Report.pdf`: Final project report and methodology summary.

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
python evaluate.py --model brnn_complex --sequence_length 200 --epochs 5 --category production
```

**Available Models:**
`cnn_base`, `cnn_complex`, `rnn_base`, `rnn_complex`, `brnn_base`, `brnn_complex`, `att_base`, `att_complex`, `ann_base`, `ann_complex`.

### 3. Feature Extraction Details (The 30 Features)
Our system utilizes 30 specific lexical features including:
- **Structural**: Length metrics, path depth, subdomain counts.
- **Statistical**: Shannon Entropy of URL/Domain, character ratios (Letter/Digit/Special).
- **Security Protocols**: IP address usage, non-standard port detection, HTTPS/HTTP status.
- **Obfuscation Markers**: URL shortener detection and character repetition counts.


## Results Summary
The system achieves peak performance using the **CNN Complex** and **BRNN Complex** models, reaching an accuracy of **97.4%**. Detailed metrics and GPU utilization plots are generated automatically in the `test_results/` directory after each run.

## Citation

If you use this work in your research, please cite the original paper:

```bibtex
@article{sahingoz2024dephides,
  title={DEPHIDES: Deep Learning Based Phishing Detection System},
  author={Sahingoz, Ozgur Koray and Bayrak, Suleyman and Bulut, Guler},
  journal={IEEE Access},
  volume={12},
  pages={11166--11179},
  year={2024},
  publisher={IEEE},
  doi={10.1109/ACCESS.2024.3352629}
}
```
## Code, Datasets and Resources

Additional codes, datasets and resources can be accessed here:
[ACCESS ZIP FILES HERE](https://drive.google.com/drive/folders/1fm0BF7NthGBoaQ2qrlfeu7sARIdL1NMK?usp=sharing)

## Implemented By

1. **[Menni Charan Sree Teja](https://github.com/charan-435)**
2. **[Ponaganti Sai Deva Charan](https://github.com/Charanponaganti)**
3. **[Lalit Santosh Deshmane](https://github.com/lalitdeshmane10-cloud)**

---
*Implementation for research into character-level phishing detection and deep sequence modeling.*

