"""
Plot score distributions per label from eval_results.json.

Usage:
    python plot_scores.py eval_results.json [--metric dice|nsd] [--output scores.png]
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="eval_results.json from eval.py")
    parser.add_argument("--metric", default="dice", choices=["dice", "nsd"])
    parser.add_argument("--output", default="scores.png")
    args = parser.parse_args()

    with open(args.results) as f:
        all_results = json.load(f)

    # collect per-label values across subjects
    label_scores = {}
    for res in all_results.values():
        if "error" in res:
            continue
        for label, metrics in res.items():
            v = metrics.get(args.metric, float("nan"))
            if not np.isnan(v):
                label_scores.setdefault(label, []).append(v)

    labels = list(label_scores.keys())
    data   = [label_scores[l] for l in labels]
    means  = [np.mean(d) for d in data]
    order  = np.argsort(means)[::-1]
    labels = [labels[i] for i in order]
    data   = [data[i]   for i in order]

    fig, ax = plt.subplots(figsize=(14, 6))
    vp = ax.violinplot(data, positions=range(len(labels)), showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_alpha(0.6)
    ax.scatter(range(len(labels)), [np.mean(d) for d in data], color="red", s=20, zorder=3, label="mean")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(args.metric.upper())
    ax.set_ylim(0, 1)
    ax.set_title(f"{args.metric.upper()} distribution per label  (n={len(all_results)} subjects)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
