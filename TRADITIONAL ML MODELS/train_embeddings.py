import os
import json
from data_loader import load_data_embeddings
from embedding_features import load_embeddings
from models import get_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "dataset")
if not os.path.isdir(DATA_ROOT):
    DATA_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "dataset"))

EMBEDDING_PATH = os.path.join(SCRIPT_DIR, "char_embeddings.json")
if not os.path.isfile(EMBEDDING_PATH):
    EMBEDDING_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "char_embeddings.json"))

if not os.path.isfile(EMBEDDING_PATH):
    raise FileNotFoundError(
        "char_embeddings.json not found. Place it in TRADITIONAL ML MODELS or the parent project folder."
    )

char_embeddings = load_embeddings(EMBEDDING_PATH)

TRAIN_PATH = os.path.join(DATA_ROOT, "train", "train.txt")
VAL_PATH = os.path.join(DATA_ROOT, "val", "val.txt")
TEST_PATH = os.path.join(DATA_ROOT, "test", "test.txt")

print(f"[train_embeddings] Using dataset root: {DATA_ROOT}")
print(f"[train_embeddings] Using embeddings file: {EMBEDDING_PATH}")

# Load data
X_train, y_train = load_data_embeddings(TRAIN_PATH, char_embeddings)
X_val, y_val = load_data_embeddings(VAL_PATH, char_embeddings)
X_test, y_test = load_data_embeddings(TEST_PATH, char_embeddings)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

embeddings_output = os.path.join(SCRIPT_DIR, "output_data", "embeddings")
os.makedirs(embeddings_output, exist_ok=True)
joblib.dump(scaler, os.path.join(embeddings_output, "embedding_scaler.joblib"))

MODEL_NAMES = ["RF", "NB", "SVM", "LR", "KNN"]

for model_name in MODEL_NAMES:
    print(f"\n🚀 Running embedding model {model_name}...")
    model = get_model(model_name)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)

    print(f"{model_name} Validation: {val_acc}")
    print(f"{model_name} Test: {test_acc}")
    print(report)

    save_dir = os.path.join(SCRIPT_DIR, "output_data", "embeddings", model_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "validation_accuracy.txt"), "w") as f:
        f.write(f"Validation Accuracy: {val_acc}\n")
    with open(os.path.join(save_dir, "test_accuracy.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc}\n")
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(save_dir, "confusion_matrix.json"), "w") as f:
        json.dump(cm.tolist(), f)

    # Save trained model
    joblib.dump(model, os.path.join(save_dir, f"{model_name}_model.joblib"))

print("\n✅ Embedding model training completed!")