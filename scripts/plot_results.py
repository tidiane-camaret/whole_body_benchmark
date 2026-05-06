"""
Plot per-label Dice / NSD distributions for one or more results JSONs.

Examples:

  python scripts/plot_results.py results/results_nnunet.json results/results_patchwork.json
  python scripts/plot_results.py results/*.json --metric nsd --output plots/nsd.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


LABEL_ORDER = [
    "AT_pelvis", "IMAT_pelvis", "Muscle_pelvis", "AVAT_pelvis",
    "AT_upper_abdomen", "IMAT_upper_abdomen", "Muscle_upper_abdomen", "AVAT_upper_abdomen",
    "AT_thorax", "IMAT_thorax", "Muscle_thorax", "AVAT_thorax",
    "heart", "IVD", "vertebra_body", "vertebra_posterior_elements",
    "liver", "pankreas", "aorta",
    "kidney_cortex_left", "kidney_hilus_left", "kidney_medulla_left",
    "kidney_cortex_right", "kidney_hilus_right", "kidney_medulla_right",
]

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def load_values(path, metric):
    with open(path) as f:
        data = json.load(f)
    per_label = {label: [] for label in LABEL_ORDER}
    for subject_res in data.values():
        if "error" in subject_res:
            continue
        for label in LABEL_ORDER:
            v = subject_res.get(label, {}).get(metric, float("nan"))
            if not np.isnan(v):
                per_label[label].append(v)
    return per_label


def plot(json_paths, metric, output):
    datasets = []
    for p in json_paths:
        name = Path(p).stem.replace("results_", "")
        datasets.append((name, load_values(p, metric)))

    n_labels = len(LABEL_ORDER)
    n_ds = len(datasets)
    width = 0.8 / n_ds
    x = np.arange(n_labels)

    fig, ax = plt.subplots(figsize=(max(18, n_labels * 0.9), 6))

    patches = []
    for i, (name, per_label) in enumerate(datasets):
        color = COLORS[i % len(COLORS)]
        offsets = (i - (n_ds - 1) / 2) * width
        positions = x + offsets
        data_by_label = [per_label[lbl] for lbl in LABEL_ORDER]

        bp = ax.boxplot(
            data_by_label,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            manage_ticks=False,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            flierprops=dict(marker="o", markersize=2, alpha=0.4, markeredgecolor=color),
            boxprops=dict(facecolor=color, alpha=0.7, color=color),
        )
        patches.append(mpatches.Patch(facecolor=color, alpha=0.7, label=name))

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric.upper())
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Per-label {metric.upper()} distribution")
    ax.legend(handles=patches, loc="lower left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsons", nargs="+", help="Results JSON files")
    parser.add_argument("--metric", default="dice", choices=["dice", "nsd"])
    parser.add_argument("--output", default="results/plots/results_distribution.png")
    args = parser.parse_args()
    plot(args.jsons, args.metric, args.output)


if __name__ == "__main__":
    main()
