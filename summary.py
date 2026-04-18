# summary.py
# reads all raw_test_results.json files under test_results/ and
# generates comparison charts across all trained models
#
# usage:
#   python summary.py
#   python summary.py --results_dir test_results

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def collect_results(results_dir):
    # walk the results directory and collect all result files
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


def plot_metrics(records, out_path):
    # bar chart comparing test accuracy across all models
    labels = [r["label"] for r in records]
    accs = [r["accuracy"] for r in records]

    # colour-code bars by category
    categories = list(dict.fromkeys(r["category"] for r in records))
    palette = plt.cm.tab10.colors
    colors = [palette[categories.index(r["category"]) % len(palette)] for r in records]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.4), 6))
    bars = ax.bar(x, accs, color=colors, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("model accuracy comparison")
    ax.grid(axis="y", alpha=0.3)

    # show accuracy value above each bar
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=7)

    # legend showing categories
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[i % len(palette)])
        for i, _ in enumerate(categories)
    ]
    ax.legend(handles, categories, title="category", fontsize=8, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_running_time(records, out_path):
    # horizontal bar chart showing how long each model took to train
    labels = [r["label"] for r in records]
    times = [r["training_time"] for r in records]

    categories = list(dict.fromkeys(r["category"] for r in records))
    palette = plt.cm.tab10.colors
    colors = [palette[categories.index(r["category"]) % len(palette)] for r in records]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.55)))
    bars = ax.barh(y, times, color=colors, edgecolor="white", linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("training time (seconds)")
    ax.set_title("model training time comparison")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    # show time value at end of each bar
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + max(times) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{t:.1f}s", va="center", fontsize=7)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[i % len(palette)])
        for i, _ in enumerate(categories)
    ]
    ax.legend(handles, categories, title="category", fontsize=8, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def main(args):
    if not os.path.isdir(args.results_dir):
        print(f"directory not found: {args.results_dir}")
        return

    records = collect_results(args.results_dir)
    if not records:
        print("no raw_test_results.json files found - run evaluate.py first")
        return

    print(f"found {len(records)} model result(s)")
    plot_metrics(records, os.path.join(args.results_dir, "metrics.png"))
    plot_running_time(records, os.path.join(args.results_dir, "running_time.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="test_results")
    args = parser.parse_args()
    main(args)