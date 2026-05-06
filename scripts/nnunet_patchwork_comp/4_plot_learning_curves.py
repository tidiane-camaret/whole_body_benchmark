"""
Plot validation learning curves comparing nnUNet and Patchwork.

Usage
-----
  python 4_plot_learning_curves.py -o <output_dir> [--nnunet <dir>] [--patchwork <dir>]

Example
-------
  python scripts/nnunet_patchwork_comp/4_plot_learning_curves.py \
    --nnunet   results/learning_curves/nnunet_all \
    --patchwork results/learning_curves/patchwork_all \
    -o results/learning_curves
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_nnunet(curve_dir: Path):
    vs = pd.read_csv(curve_dir / "valid_scalar.csv", index_col="epoch")
    return vs["walltime_h"], vs["loss"], vs["mean_dice"]


def load_patchwork(curve_dir: Path):
    vs  = pd.read_csv(curve_dir / "valid_scalar.csv",   index_col="step")
    vpc = pd.read_csv(curve_dir / "valid_perclass.csv", index_col="step")
    class_cols = [c for c in vpc.columns
                  if c not in ("walltime_s", "walltime_h", "walltime_extrapolated")]
    mean_f1 = vpc[class_cols].mean(axis=1)
    return vs["walltime_h"], vs["loss"], mean_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnunet",    type=Path, default=Path("results/learning_curves/nnunet_all"))
    parser.add_argument("--patchwork", type=Path, default=Path("results/learning_curves/patchwork_all"))
    parser.add_argument("-o", "--output_dir", type=Path, default=Path("results/learning_curves"))
    args = parser.parse_args()

    nn_t,  nn_loss,  nn_dice  = load_nnunet(args.nnunet)
    pw_t,  pw_loss,  pw_dice  = load_patchwork(args.patchwork)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Dice / F1 ---
    ax = axes[0]
    ax.plot(nn_t, nn_dice, label="nnUNet (mean pseudo-dice)", color="steelblue", linewidth=1.5)
    ax.plot(pw_t, pw_dice, label="Patchwork (mean val F1)",   color="darkorange", linewidth=1.5)
    ax.set_xlabel("Wall time (h)")
    ax.set_ylabel("Mean Dice / F1")
    ax.set_title("Validation Dice over time")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Loss ---
    ax = axes[1]
    ax.plot(nn_t, nn_loss, label="nnUNet val loss",    color="steelblue", linewidth=1.5)
    ax.plot(pw_t, pw_loss, label="Patchwork val loss", color="darkorange", linewidth=1.5)
    ax.set_xlabel("Wall time (h)")
    ax.set_ylabel("Loss")
    ax.set_title("Validation loss over time\n(different scales — convergence shape only)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = args.output_dir / "val_curves_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
