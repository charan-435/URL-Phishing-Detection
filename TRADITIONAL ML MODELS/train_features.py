from feature_extraction import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import os
import json
import seaborn as sns
import matplotlib.pyplot as plt

# -------- LOAD DATA -------- #

fe = FeatureExtractor()

fe.load_from_file("../dataset/train/train.txt")
X_train = fe.get_handcrafted()
y_train = fe.get_labels()

fe.load_from_file("../dataset/test/test.txt")
X_test = fe.get_handcrafted()
y_test = fe.get_labels()

# -------- MODELS -------- #

models = {
    "RF": RandomForestClassifier(n_estimators=100),
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

    save_dir = f"results/{name}"
    os.makedirs(save_dir, exist_ok=True)

    # Accuracy
    with open(f"{save_dir}/accuracy.txt", "w") as f:
        f.write(f"Accuracy: {acc}")

    # Report
    with open(f"{save_dir}/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix JSON
    with open(f"{save_dir}/confusion_matrix.json", "w") as f:
        json.dump(cm.tolist(), f)

    # Confusion matrix plot
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["phishing","legitimate"],
                yticklabels=["phishing","legitimate"])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.close()

print("\n✅ All models completed!")