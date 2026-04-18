# URL-Phishing-Detection (DEPHIDES Implementation)

Welcome to the **URL-Phishing-Detection** project! This is a deep-learning-based system inspired by the **DEPHIDES** research paper. It uses 5 different neural network architectures to identify malicious URLs with high accuracy.

## 🚀 Project Overview
The goal of this project is to compare how different deep learning models handle phishing detection. We look at URLs in two ways:
1.  **Character Sequences**: Treating the URL as a string of characters (like a sentence).
2.  **Hand-crafted Features**: Using specific metrics like URL length, dot count, and entropy.

## 📁 System Structure
- `models/`: Contains the architecture for all 5 model types:
    - `cnn/`: Convolutional Neural Networks (Best for local patterns).
    - `rnn/`: Recurrent Neural Networks (Best for sequential data).
    - `brnn/`: Bidirectional RNNs.
    - `att/`: Attention-based Networks.
    - `ann/`: Standard Artificial Neural Networks.
- `dataset/`: Contains the training and testing data.
- `evaluate.py`: The main script to train and test any model.
- `summary.py`: Generates comparison charts once you've trained your models.
- `feature_extraction.py`: The logic for turning raw URLs into model-ready data.

## 🛠️ How to Run

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Train a Model
Use `evaluate.py` to train any model from the `models/` folder. For example:
```bash
# To train the CNN Base model for 10 epochs
python evaluate.py --model cnn_base --epochs 10

# To train the ANN using hand-crafted features
python evaluate.py --model ann_base --epochs 20
```

### 3. See the Results
After training your models, run the summary script to generate performance charts:
```bash
python summary.py
```
This will create `metrics.png` and `running_time.png` in the `test_results/` folder.

## 📊 Models in this Project
- **Base Models**: Simpler, faster architectures for quick testing.
- **Complex Models**: Deeper networks designed for maximum accuracy (~98% in the paper).

---
*Developed as part of a research project on Deep Learning Based Phishing Detection.*