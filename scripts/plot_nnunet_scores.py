"""
Plot Dice distribution per label from an nnUNet summary.json.

Usage:
    python plot_nnunet_scores.py summary.json dataset.json [--metric dice|iou] [--output scores_nnunet.png]
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", help="summary.json from nnUNet evaluation")
    parser.add_argument("dataset", help="dataset.json with label name mapping")
    parser.add_argument("--metric", default="dice", choices=["dice", "iou"])
    parser.add_argument("--output", default="scores_nnunet.png")
    args = parser.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)
    with open(args.dataset) as f:
        ds = json.load(f)

    metric_key = "Dice" if args.metric == "dice" else "IoU"

    # invert label map: int_id -> name
    id2name = {str(v): k for k, v in ds["labels"].items() if k != "background"}

    # collect per-label scores across cases
    label_scores = {}
    for case in summary["metric_per_case"]:
        for label_id, metrics in case["metrics"].items():
            name = id2name.get(label_id, f"label_{label_id}")
            v = metrics.get(metric_key, float("nan"))
            if not np.isnan(v):
                label_scores.setdefault(name, []).append(v)

    labels = list(label_scores.keys())
    data   = [label_scores[l] for l in labels]
    means  = [np.mean(d) for d in data]
    order  = np.argsort(means)[::-1]
    labels = [labels[i] for i in order]
    data   = [data[i]   for i in order]

    n_cases = len(summary["metric_per_case"])
    fig, ax = plt.subplots(figsize=(14, 6))
    vp = ax.violinplot(data, positions=range(len(labels)), showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_alpha(0.6)
    ax.scatter(range(len(labels)), [np.mean(d) for d in data],
               color="red", s=20, zorder=3, label="mean")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(args.metric.upper())
    ax.set_ylim(0, 1)
    ax.set_title(f"nnUNet {args.metric.upper()} distribution per label  (n={n_cases} subjects)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
