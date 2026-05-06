"""
Extract nnUNet learning curves from a training log directory.

Outputs (parallel structure to 3_extract_learning_curve_patchwork.py):
  - train_scalar.csv   : epoch, walltime_s, walltime_h, loss
  - valid_scalar.csv   : epoch, walltime_s, walltime_h, loss, mean_dice
  - valid_perclass.csv : epoch, walltime_s, walltime_h, 0..N-1 (per-class pseudo dice)

Notes on metrics
----------------
- loss: nnUNet compound loss (CE + Dice), lower is better (can be negative)
- mean_dice: mean of per-class pseudo dice logged at end of each epoch
- pseudo dice is computed on the val loader with fixed batch sampling (not full volumes)
- no per-class train dice available in nnUNet logs

Usage
-----
  python 3_extract_learning_curve_nnunet.py <result_dir> [-o <output_dir>]

Example
-------
  python 3_extract_learning_curve_nnunet.py \\
    /nfs/.../nnUNetTrainer__nnUNetResEncUNetPlans_40G__3d_fullres/fold_all \\
    -o results/learning_curves/nnunet_all
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


def load_nnunet_curves(result_dir: Path):
    log_files = sorted(result_dir.glob("training_log_*.txt"))
    if not log_files:
        raise FileNotFoundError(f"No training_log_*.txt found in {result_dir}")
    # Concatenate all log files in chronological order (named by date)
    log_text = "\n".join(p.read_text() for p in log_files)

    epoch_starts = re.findall(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+): Epoch (\d+)\s*\n",
        log_text,
    )
    train_losses = [float(x) for x in re.findall(r"train_loss (-?[0-9.]+)", log_text)]
    val_losses   = [float(x) for x in re.findall(r"val_loss (-?[0-9.]+)", log_text)]
    dice_rows    = [
        [float(v) for v in row.split(", ")]
        for row in re.findall(r"Pseudo dice \[([0-9., ]+)\]", log_text)
    ]

    n = min(len(train_losses), len(val_losses), len(dice_rows))
    if n == 0:
        raise ValueError("No complete epoch data found in log.")

    # Epoch N's metrics are logged between marker[N] and marker[N+1].
    # Use marker[N+1] as the wall-clock time at which epoch N completed.
    timestamps = [
        datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
        for ts, _ in epoch_starts
    ]
    t0 = timestamps[0]

    def walltime_s(epoch_idx: int) -> float:
        # Use next marker if available, otherwise extrapolate from last interval
        if epoch_idx + 1 < len(timestamps):
            return (timestamps[epoch_idx + 1] - t0).total_seconds()
        elif len(timestamps) >= 2:
            mean_epoch_s = (timestamps[-1] - t0).total_seconds() / (len(timestamps) - 1)
            return (timestamps[-1] - t0).total_seconds() + mean_epoch_s
        return 0.0

    epochs = list(range(n))
    wt_s   = [walltime_s(i) for i in epochs]
    wt_h   = [s / 3600.0 for s in wt_s]

    train_df = pd.DataFrame({
        "walltime_s": wt_s,
        "walltime_h": wt_h,
        "loss": train_losses[:n],
    }, index=pd.Index(epochs, name="epoch"))

    mean_dice = [sum(row) / len(row) for row in dice_rows[:n]]
    valid_df = pd.DataFrame({
        "walltime_s": wt_s,
        "walltime_h": wt_h,
        "loss":       val_losses[:n],
        "mean_dice":  mean_dice,
    }, index=pd.Index(epochs, name="epoch"))

    valid_pc = pd.DataFrame(
        dice_rows[:n],
        index=pd.Index(epochs, name="epoch"),
    )

    meta = {
        "n_epochs_logged": n,
        "n_epochs_total":  int(re.search(r"num_epochs.*?(\d+)", log_text).group(1))
                           if re.search(r"num_epochs.*?(\d+)", log_text) else None,
        "log_files":       [p.name for p in log_files],
    }
    return train_df, valid_df, valid_pc, meta


def main():
    parser = argparse.ArgumentParser(description="Extract nnUNet learning curves to CSV.")
    parser.add_argument("result_dir", type=Path,
                        help="nnUNet fold directory containing training_log_*.txt")
    parser.add_argument("-o", "--output_dir", type=Path, default=None,
                        help="Output directory (default: result_dir)")
    args = parser.parse_args()

    out_dir = args.output_dir or args.result_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, valid_pc, meta = load_nnunet_curves(args.result_dir)

    train_df.to_csv(out_dir / "train_scalar.csv")
    valid_df.to_csv(out_dir / "valid_scalar.csv")
    valid_pc.to_csv(out_dir / "valid_perclass.csv")

    print(f"Wrote 3 CSV files to {out_dir}")
    print(f"  Epochs:      0 → {len(train_df) - 1}  ({len(train_df)} logged"
          + (f" / {meta['n_epochs_total']} planned)" if meta["n_epochs_total"] else ")"))
    print(f"  Wall time:   {valid_df['walltime_h'].iloc[-1]:.1f} h")
    print(f"  Final val loss:      {valid_df['loss'].iloc[-1]:.4f}")
    print(f"  Final val mean dice: {valid_df['mean_dice'].iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
