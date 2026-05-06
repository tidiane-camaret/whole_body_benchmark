"""
Extract patchwork learning curves from a result directory.

Outputs:
  - train_scalar.csv   : step, walltime_s, walltime_h, loss, f1
  - valid_scalar.csv   : step, walltime_s, walltime_h, loss, f1
  - train_perclass.csv : step, walltime_s, walltime_h, 0..25 (per-class F1)
  - valid_perclass.csv : step, walltime_s, walltime_h, 0..25 (per-class F1)

Notes on metrics
----------------
- f1 = Dice with Laplace smoothing: (2*TP+1) / (possible_pos + pred_pos + 2)
- validation f1 uses oracle threshold (best over all thresholds) → optimistic vs nnUNet
- train f1 uses fixed 0.5 threshold
- step = cumulative number of patches seen during training

Usage
-----
  python 3_extract_learning_curve_patchwork.py <result_dir> [-o <output_dir>]

Example
-------
  python 3_extract_learning_curve_patchwork.py \
    /nfs/data/nii/.../patchwork_oppscreen_all \
    -o ./curves/patchwork
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_patchwork_curves(result_dir: Path):
    with open(result_dir / "model_patchwork.json") as f:
        d = json.load(f)

    th = d["trainloss_hist"]
    vh = d["validloss_hist"]

    def scalar_series(hist_dict, key):
        steps, vals = zip(*hist_dict[key])
        return pd.Series(vals, index=list(steps), name=key)

    def perclass_df(hist_dict, key):
        entries = hist_dict[key]
        steps = [e[0] for e in entries]
        vals  = [e[1] for e in entries]
        return pd.DataFrame(vals, index=steps)

    train_df = pd.DataFrame({
        "loss": scalar_series(th, "output_4_loss"),
        "f1":   scalar_series(th, "output_4_f1"),
    })
    has_valid = bool(vh) and "valid_output_4_loss" in vh
    if has_valid:
        valid_df = pd.DataFrame({
            "loss": scalar_series(vh, "valid_output_4_loss"),
            "f1":   scalar_series(vh, "valid_output_4_f1"),
        })
        valid_perclass = perclass_df(vh, "valid_nodisplay_class_f1")
    else:
        valid_df = pd.DataFrame(columns=["loss", "f1"])
        valid_perclass = pd.DataFrame()
    train_perclass = perclass_df(th, "nodisplay_class_f1")

    # Wall time from trainlog.txt. Each log entry covers one training cycle
    # (N_epochs × step_size patches). Expand each entry into per-checkpoint
    # slots, then pad any remainder with the mean per-checkpoint duration.
    log_text = (result_dir / "trainlog.txt").read_text()
    elapsed  = [float(x) for x in re.findall(r"time elapsed, fitting: ([0-9.]+)", log_text)]

    step_size  = int(train_df.index[1] - train_df.index[0])
    n_iters    = int(train_df.index[-1] // step_size) + 1

    # How many checkpoint steps does each log entry cover?
    # (e.g. 4 epochs × step_size patches per epoch = 4 checkpoints per entry)
    steps_per_entry = max(1, round(n_iters / len(elapsed))) if elapsed else 1
    # Expand: each log entry contributes equally to its covered checkpoints
    elapsed_per_step = [t / steps_per_entry for t in elapsed for _ in range(steps_per_entry)]
    mean_step = float(np.mean(elapsed_per_step)) if elapsed_per_step else 0.0

    # Pad to full length if log was truncated
    elapsed_full = elapsed_per_step + [mean_step] * max(0, n_iters - len(elapsed_per_step))
    cumtime = np.cumsum([0.0] + elapsed_full)  # index i = wall seconds after checkpoint i

    n_logged = len(elapsed_per_step)  # number of checkpoints actually covered by the log

    def add_walltime(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index.name = "step"
        iters = df.index // step_size
        df.insert(0, "walltime_s", [float(cumtime[i]) for i in iters])
        df.insert(1, "walltime_h", df["walltime_s"] / 3600.0)
        df.insert(2, "walltime_extrapolated", iters >= n_logged)
        return df

    meta = {
        "n_iter_logged":    n_logged,
        "n_iter_total":     n_iters,
        "mean_iter_s":      mean_step,
        "steps_per_entry":  steps_per_entry,
    }
    return (
        add_walltime(train_df),
        add_walltime(valid_df),
        add_walltime(train_perclass),
        add_walltime(valid_perclass),
        meta,
    )


def main():
    parser = argparse.ArgumentParser(description="Extract patchwork learning curves to CSV.")
    parser.add_argument("result_dir", type=Path, help="Patchwork result directory containing model_patchwork.json")
    parser.add_argument("-o", "--output_dir", type=Path, default=None,
                        help="Output directory (default: result_dir)")
    args = parser.parse_args()

    out_dir = args.output_dir or args.result_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, train_pc, valid_pc, meta = load_patchwork_curves(args.result_dir)

    train_df.to_csv(out_dir / "train_scalar.csv")
    valid_df.to_csv(out_dir / "valid_scalar.csv")
    train_pc.to_csv(out_dir / "train_perclass.csv")
    valid_pc.to_csv(out_dir / "valid_perclass.csv")

    print(f"Wrote 4 CSV files to {out_dir}")
    print(f"  Steps:       {train_df.index[0]} → {train_df.index[-1]}  ({len(train_df)} checkpoints)")
    if not valid_df.empty:
        n_extrap = int(valid_df["walltime_extrapolated"].sum())
        extrap_note = (f"  ({n_extrap} checkpoints extrapolated from mean iter "
                       f"{meta['mean_iter_s']:.0f}s)") if n_extrap else ""
        print(f"  Wall time:   {valid_df['walltime_h'].iloc[-1]:.1f} h{extrap_note}")
        print(f"  Final valid f1: {valid_df['f1'].iloc[-1]:.4f}")
    else:
        n_extrap = int(train_df["walltime_extrapolated"].sum())
        extrap_note = (f"  ({n_extrap} checkpoints extrapolated from mean iter "
                       f"{meta['mean_iter_s']:.0f}s)") if n_extrap else ""
        print(f"  Wall time:   {train_df['walltime_h'].iloc[-1]:.1f} h{extrap_note}")
        print("  No validation data.")


if __name__ == "__main__":
    main()
