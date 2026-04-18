# evaluate.py
# this is the main training + evaluation script
# it trains a chosen model on the URL dataset and saves all the results
#
# usage examples:
#   python evaluate.py --category baseline     --model cnn_base    --epochs 10
#   python evaluate.py --category big_dataset  --model cnn_complex --epochs 20
#   python evaluate.py --category complex_arch --model att_complex --epochs 15

import argparse
import json
import os
import sys
import time
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_extraction import FeatureExtractor


# ---- model registry ----
# maps model name strings to their builder classes

def get_model_builder(model_name, embed_dim, seq_len):
    from models.cnn.cnn_base    import CnnBase
    from models.cnn.cnn_complex import CnnComplex
    from models.rnn.rnn_base    import RnnBase
    from models.rnn.rnn_complex import RnnComplex
    from models.brnn.brnn_base    import BrnnBase
    from models.brnn.brnn_complex import BrnnComplex
    from models.att.att_base    import AttBase
    from models.att.att_complex import AttComplex
    from models.ann.ann_base    import AnnBase
    from models.ann.ann_complex import AnnComplex

    models = {
        "cnn_base":     CnnBase,
        "cnn_complex":  CnnComplex,
        "rnn_base":     RnnBase,
        "rnn_complex":  RnnComplex,
        "brnn_base":    BrnnBase,
        "brnn_complex": BrnnComplex,
        "att_base":     AttBase,
        "att_complex":  AttComplex,
        "ann_base":     AnnBase,
        "ann_complex":  AnnComplex,
    }

    if model_name not in models:
        raise ValueError(f"unknown model '{model_name}'. options are: {list(models.keys())}")
    
    # ann_base is special because it uses hand-crafted features, not embeddings
    if model_name == "ann_base":
        # we'll pass the feature count later or handle it here
        # for now, let's just return the class or a lambda
        return AnnBase
    
    return models[model_name](embed_dim, seq_len)


# ---- gpu monitor (runs in background thread during training) ----

class GpuMonitor:
    # polls nvidia-smi every second to track gpu usage %

    def __init__(self):
        self.readings = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _poll(self):
        import subprocess
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                self.readings.append(float(out.split("\n")[0]))
            except Exception:
                self.readings.append(0.0)  # no gpu, just log 0
            time.sleep(1)

    def save_plot(self, path):
        fig, ax = plt.subplots(figsize=(10, 4))
        if self.readings:
            ax.plot(self.readings, color="green", linewidth=1.2)
            ax.fill_between(range(len(self.readings)), self.readings, alpha=0.25, color="green")
            ax.set_ylim(0, 100)
        else:
            ax.text(0.5, 0.5, "no gpu detected", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("gpu utilisation (%)")
        ax.set_title("gpu usage during training")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


# ---- plotting helpers ----

def plot_accuracy(history, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["accuracy"],     label="train accuracy",      linewidth=1.5)
    ax.plot(history.history["val_accuracy"], label="validation accuracy", linewidth=1.5, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("model accuracy over epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss(history, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"],     label="train loss",      linewidth=1.5)
    ax.plot(history.history["val_loss"], label="validation loss", linewidth=1.5, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("model loss over epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, labels, path, normalize=False):
    title = "normalized confusion matrix" if normalize else "confusion matrix"
    vals = cm.astype("float") / cm.sum(axis=1, keepdims=True) if normalize else cm
    fmt = ".2f" if normalize else "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=vals, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, values_format=fmt)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- main ----

def main(args):
    out_dir = os.path.join("test_results", args.category, args.model)
    os.makedirs(out_dir, exist_ok=True)
    print(f"saving results to: {out_dir}")

    # save the exact command used to run this so we can reproduce it later
    with open(os.path.join(out_dir, "floyd_command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    # load training data
    fe = FeatureExtractor(char_index_path="dataset/char_index")
    fe.load_from_file("dataset/train/train.txt")
    
    if args.model == "ann_base":
        print("using hand-crafted features for ann_base")
        x_train = fe.get_handcrafted()
    else:
        x_train = fe.get_sequences(sequence_length=args.sequence_length)
    
    y_train = fe.get_labels()

    # load test data
    fe_test = FeatureExtractor(char_index_path="dataset/char_index")
    fe_test.load_from_file("dataset/test/test.txt")
    
    if args.model == "ann_base":
        x_test = fe_test.get_handcrafted()
    else:
        x_test = fe_test.get_sequences(sequence_length=args.sequence_length)
    
    y_test = fe_test.get_labels()

    # save the character embedding map
    char_index = fe.tokener.word_index
    with open(os.path.join(out_dir, "char_embeddings.json"), "w") as f:
        json.dump(char_index, f, indent=2)

    # build and compile the model
    builder_or_class = get_model_builder(args.model, args.embed_dim, args.sequence_length)
    
    if args.model == "ann_base":
        # instantiate with feature count
        model = builder_or_class(feature_count=x_train.shape[1]).build()
    else:
        model = builder_or_class.build(char_index)
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    # save the model architecture summary to a text file
    with open(os.path.join(out_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    # save model architecture as json
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        f.write(model.to_json(indent=2))

    # start gpu monitoring in background
    gpu_mon = GpuMonitor()
    gpu_mon.start()
    t_start = time.time()

    # train the model
    history = model.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        verbose=1,
    )

    t_end = time.time()
    gpu_mon.stop()
    train_time = t_end - t_start

    # save weights and full model
    model.save_weights(os.path.join(out_dir, "weights.weights.h5"))
    model.save(os.path.join(out_dir, "model_all.keras"))

    # generate training curve plots
    plot_accuracy(history, os.path.join(out_dir, "accuracy.png"))
    plot_loss(history, os.path.join(out_dir, "loss.png"))
    gpu_mon.save_plot(os.path.join(out_dir, "gpu_utilization.png"))

    # run on test set
    print("running inference on test set...")
    y_prob = model.predict(x_test, batch_size=args.batch_size, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    labels = ["phishing", "legitimate"]

    # confusion matrix plots
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, labels, os.path.join(out_dir, "confusion_matrix.png"), normalize=False)
    plot_confusion(cm, labels, os.path.join(out_dir, "normalized_confusion_matrix.png"), normalize=True)

    # classification report
    report = classification_report(y_test, y_pred, target_names=labels)
    print(report)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # dump raw results to json for later analysis
    raw_results = {
        "model":           args.model,
        "category":        args.category,
        "epochs":          args.epochs,
        "batch_size":      args.batch_size,
        "embed_dim":       args.embed_dim,
        "sequence_length": args.sequence_length,
        "training_time_seconds": round(train_time, 2),
        "test_accuracy":   float(np.mean(y_pred == y_test)),
        "y_true":          y_test.tolist(),
        "y_pred":          y_pred.tolist(),
        "y_pred_prob":     y_prob.tolist(),
        "history": {
            "accuracy":     history.history["accuracy"],
            "val_accuracy": history.history["val_accuracy"],
            "loss":         history.history["loss"],
            "val_loss":     history.history["val_loss"],
        },
    }
    with open(os.path.join(out_dir, "raw_test_results.json"), "w") as f:
        json.dump(raw_results, f, indent=2)

    print(f"all results saved to: {out_dir}")
    print(f"test accuracy: {raw_results['test_accuracy']:.4f}")
    print(f"training time: {train_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train and evaluate a url phishing detection model")
    parser.add_argument("--category",        type=str, default="baseline",
                        choices=["baseline", "big_dataset", "complex_arch", "complex_cnn", "traditional_ml"])
    parser.add_argument("--model",           type=str, default="cnn_base")
    parser.add_argument("--embed_dim",       type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--epochs",          type=int, default=10)
    parser.add_argument("--batch_size",      type=int, default=32)
    args = parser.parse_args()

    main(args)