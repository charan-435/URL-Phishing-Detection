from feature_extraction import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "dataset")
if not os.path.isdir(DATA_ROOT):
    DATA_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "dataset"))

# -------- LOAD DATA -------- #

fe = FeatureExtractor()

fe.load_from_file(os.path.join(DATA_ROOT, "train", "train.txt"))
X_train = fe.get_handcrafted()
y_train = fe.get_labels()

fe.load_from_file(os.path.join(DATA_ROOT, "test", "test.txt"))
X_test = fe.get_handcrafted()
y_test = fe.get_labels()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

output_root = os.path.join(SCRIPT_DIR, "output_data")
os.makedirs(output_root, exist_ok=True)
joblib.dump(scaler, os.path.join(output_root, "feature_scaler.joblib"))

# -------- MODELS -------- #

models = {
   # "RF": RandomForestClassifier(n_estimators=100),
    "SVM": svm.SVC(),
    "LR": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NB": GaussianNB()
}

# -------- RUN ALL MODELS -------- #

for name, model in models.items():
    print(f"\n🚀 Running {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"{name} Accuracy: {acc}")

    # -------- SAVE RESULTS -------- #

    save_dir = os.path.join(SCRIPT_DIR, "output_data", name)
    os.makedirs(save_dir, exist_ok=True)

    # Accuracy
    with open(os.path.join(save_dir, "accuracy.txt"), "w") as f:
        f.write(f"Accuracy: {acc}")

    # Report
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix JSON
    with open(os.path.join(save_dir, "confusion_matrix.json"), "w") as f:
        json.dump(cm.tolist(), f)

    # Save trained model
    joblib.dump(model, os.path.join(save_dir, f"{name}_model.joblib"))

    # Confusion matrix plot
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["phishing","legitimate"],
                yticklabels=["phishing","legitimate"])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

print("\n✅ All models completed!")