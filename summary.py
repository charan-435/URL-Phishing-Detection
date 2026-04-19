# summary script to compare all results
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# find all json results and collect them
def get_results(res_dir):
    data_list = []
    for cat in sorted(os.listdir(res_dir)):
        cat_path = os.path.join(res_dir, cat)
        if not os.path.isdir(cat_path): continue
        for model_name in sorted(os.listdir(cat_path)):
            json_file = os.path.join(cat_path, model_name, "raw_test_results.json")
            if not os.path.isfile(json_file): continue
            with open(json_file) as f:
                res = json.load(f)
            data_list.append({
                "label": f"{cat}\n{model_name}",
                "cat": cat, "name": model_name,
                "acc": res.get("test_accuracy", 0.0),
                "time": res.get("training_time_seconds", 0.0)
            })
    return data_list

# accuracy chart
def plot_acc(results, path):
    names = [r["label"] for r in results]
    accs = [r["acc"] for r in results]
    cats = list(dict.fromkeys(r["cat"] for r in results))
    colors = plt.cm.tab10.colors
    bar_colors = [colors[cats.index(r["cat"]) % len(colors)] for r in results]

    plt.figure(figsize=(max(10, len(names)*1.4), 6))
    bars = plt.bar(range(len(names)), accs, color=bar_colors)
    plt.xticks(range(len(names)), names, fontsize=8)
    plt.ylabel("accuracy")
    plt.title("model accuracy comparison")
    
    for bar, val in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=7)
    
    plt.savefig(path); plt.close()

# training time chart
def plot_time(results, path):
    names = [r["label"] for r in results]
    times = [r["time"] for r in results]
    cats = list(dict.fromkeys(r["cat"] for r in results))
    colors = plt.cm.tab10.colors
    bar_colors = [colors[cats.index(r["cat"]) % len(colors)] for r in results]

    plt.figure(figsize=(10, max(5, len(names)*0.6)))
    plt.barh(range(len(names)), times, color=bar_colors)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.xlabel("time (s)")
    plt.title("model training time")
    plt.gca().invert_yaxis()
    plt.savefig(path); plt.close()

def run_summary(args):
    if not os.path.isdir(args.results_dir):
        print("results dir not found!")
        return
    res = get_results(args.results_dir)
    if not res:
        print("no results found!")
        return
    plot_acc(res, os.path.join(args.results_dir, "metrics.png"))
    plot_time(res, os.path.join(args.results_dir, "running_time.png"))
    print("charts saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="test_results")
    run_summary(parser.parse_args())