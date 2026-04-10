"""
summary.py
----------
Reads all raw_test_results.json files under test_results/ and produces:
  - test_results/metrics.png      (accuracy comparison across all models)
  - test_results/running_time.png (training time comparison across all models)

Usage:
  python summary.py
  python summary.py --results_dir test_results
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def collect_results(results_dir: str) -> list:
    """Walk results_dir and collect all raw_test_results.json entries."""
    records = []
    for category in sorted(os.listdir(results_dir)):
        cat_path = os.path.join(results_dir, category)
        if not os.path.isdir(cat_path):
            continue
        for model_name in sorted(os.listdir(cat_path)):
            json_path = os.path.join(cat_path, model_name, "raw_test_results.json")
            if not os.path.isfile(json_path):
                continue
            with open(json_path) as f:
                data = json.load(f)
            records.append({
                "label":         f"{category}\n{model_name}",
                "category":      category,
                "model":         model_name,
                "accuracy":      data.get("test_accuracy", 0.0),
                "training_time": data.get("training_time_seconds", 0.0),
            })
    return records


def plot_metrics(records: list, out_path: str):
    """Bar chart of test accuracy for all models."""
    labels   = [r["label"]   for r in records]
    accuracy = [r["accuracy"] for r in records]

    # colour by category
    categories = list(dict.fromkeys(r["category"] for r in records))
    palette    = plt.cm.tab10.colors
    colors     = [palette[categories.index(r["category"]) % len(palette)] for r in records]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.4), 6))
    bars = ax.bar(x, accuracy, color=colors, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=7)

    # legend for categories
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[i % len(palette)])
        for i, _ in enumerate(categories)
    ]
    ax.legend(handles, categories, title="Category", fontsize=8,
              loc="lower right", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[summary] Saved: {out_path}")


def plot_running_time(records: list, out_path: str):
    """Horizontal bar chart of training time for all models."""
    labels = [r["label"]         for r in records]
    times  = [r["training_time"] for r in records]

    categories = list(dict.fromkeys(r["category"] for r in records))
    palette    = plt.cm.tab10.colors
    colors     = [palette[categories.index(r["category"]) % len(palette)] for r in records]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.55)))
    bars = ax.barh(y, times, color=colors, edgecolor="white", linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Model Training Time Comparison")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + max(times) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{t:.1f}s", va="center", fontsize=7)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[i % len(palette)])
        for i, _ in enumerate(categories)
    ]
    ax.legend(handles, categories, title="Category", fontsize=8,
              loc="lower right", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[summary] Saved: {out_path}")


def main(args):
    if not os.path.isdir(args.results_dir):
        print(f"[summary] Directory not found: {args.results_dir}")
        return

    records = collect_results(args.results_dir)
    if not records:
        print("[summary] No raw_test_results.json files found. Run evaluate.py first.")
        return

    print(f"[summary] Found {len(records)} model result(s).")

    plot_metrics(     records, os.path.join(args.results_dir, "metrics.png"))
    plot_running_time(records, os.path.join(args.results_dir, "running_time.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="test_results")
    args = parser.parse_args()
    main(args)