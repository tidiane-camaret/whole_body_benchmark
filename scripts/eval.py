"""
Evaluate whole-body segmentation predictions against GT.

Paths use {subject} as a placeholder.

Examples:

  # nnUNet
python scripts/eval.py \
    --gt-pattern   /nfs/data/nii/data1/Analysis/camaret___whole_body_benchmark/ANALYSIS_ana001/nnunet/nnUNet_raw/Dataset001_oppscreen_all/labelsTs/{subject}.nii.gz \
    --pred-pattern /nfs/data/nii/data1/Analysis/camaret___whole_body_benchmark/ANALYSIS_ana001/nnunet/nnUNet_raw/Dataset001_oppscreen_all/predsTs/{subject}.nii.gz \
    --subjects     /nfs/data/nii/data1/Analysis/camaret___whole_body_benchmark/ANALYSIS_ana001/nnunet/nnUNet_raw/Dataset001_oppscreen_all/labelsTs \
    --output       results/results_nnunet.json

  # Patchwork (original preds)
  python scripts/eval.py \
    --gt-pattern   /nfs/data/nii/data0/GNC/GNC_759/data/{subject}/30/opportunistic-screening/seg.nii.gz \
    --pred-pattern /nfs/data/nii/data0/GNC/GNC_759/data/{subject}/30/wholebody/subsetFW.nii.gz \
    --subjects     /nfs/data/nii/data1/Analysis/camaret___whole_body_benchmark/ANALYSIS_ana001/data/oppscreen/splits_966.json --splits-key test \
    --output       results/results_patchwork_original.json


  # Patchwork (our preds)
  python scripts/eval.py \
    --gt-pattern   /nfs/data/nii/data1/camaret___whole_body_benchmark/{subject}/NII/{subject}_mask.nii.gz \
    --pred-pattern /nfs/data/nii/data1/camaret___whole_body_benchmark/{subject}/NII/patchwork_pred_all.nii.gz \
    --subjects     /nfs/data/nii/data1/Analysis/camaret___whole_body_benchmark/ANALYSIS_ana001/data/oppscreen/splits_966.json --splits-key test \
    --output       results/results_patchwork.json
"""

import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from compute_metrics import dice, nsd

LABEL_MAP = {
    "AT_pelvis": 1,
    "IMAT_pelvis": 2,
    "Muscle_pelvis": 3,
    "AVAT_pelvis": 4,
    "AT_upper_abdomen": 5,
    "IMAT_upper_abdomen": 6,
    "Muscle_upper_abdomen": 7,
    "AVAT_upper_abdomen": 8,
    "AT_thorax": 9,
    "IMAT_thorax": 10,
    "Muscle_thorax": 11,
    "AVAT_thorax": 12,
    "heart": 13,
    "IVD": 14,
    "vertebra_body": 15,
    "vertebra_posterior_elements": 16,
    "liver": 17,
    "pankreas": 18,
    "aorta": 19,
    "kidney_cortex_left": 20,
    "kidney_hilus_left": 21,
    "kidney_medulla_left": 22,
    "kidney_cortex_right": 23,
    "kidney_hilus_right": 24,
    "kidney_medulla_right": 25,
}


def eval_subject(subject, gt_pattern, pred_pattern, compute_nsd=True, tolerance_mm=2.0):
    gt_path   = gt_pattern.format(subject=subject)
    pred_path = pred_pattern.format(subject=subject)

    try:
        gt_nii   = nib.load(gt_path)
        pred_nii = nib.load(pred_path)
    except Exception as e:
        return subject, {"error": str(e)}

    gt   = np.asarray(gt_nii.dataobj, dtype=np.int32)
    pred = np.asarray(pred_nii.dataobj, dtype=np.int32)
    if pred.ndim == 4:
        pred = pred[..., 0]

    if pred.shape != gt.shape:
        zoom_factors = tuple(g / p for g, p in zip(gt.shape, pred.shape))
        pred = zoom(pred, zoom_factors, order=0)

    voxel_spacing = gt_nii.header.get_zooms()[:3]
    results = {}
    for label_name, k in LABEL_MAP.items():
        gt_k   = gt   == k
        pred_k = pred == k
        d = dice(gt_k, pred_k)
        entry = {"dice": float(d)}
        if compute_nsd:
            entry["nsd"] = float(nsd(gt_k, pred_k, voxel_spacing, tolerance_mm))
        results[label_name] = entry

    return subject, results


def load_subjects(subjects_arg, splits_key):
    p = Path(subjects_arg)
    if p.is_dir():
        # discover subjects from directory: strip extension(s) from filenames
        return sorted(f.name.replace(".nii.gz", "").replace(".nii", "") for f in p.glob("*.nii*"))
    if p.suffix == ".json":
        with open(p) as f:
            data = json.load(f)
        if splits_key:
            return data[splits_key]
        if isinstance(data, list):
            return data
        # single-key JSON
        keys = [k for k in data if k != "all"]
        if len(keys) == 1:
            return data[keys[0]]
        raise ValueError(f"Ambiguous splits JSON — specify --splits-key. Keys: {list(data.keys())}")
    # plain text file, one subject per line
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def print_summary(all_results, compute_nsd):
    label_dice = {name: [] for name in LABEL_MAP}
    label_nsd  = {name: [] for name in LABEL_MAP} if compute_nsd else {}

    for res in all_results.values():
        if "error" in res:
            continue
        for label_name, metrics in res.items():
            v = metrics["dice"]
            if not np.isnan(v):
                label_dice[label_name].append(v)
            if compute_nsd:
                v = metrics.get("nsd", float("nan"))
                if not np.isnan(v):
                    label_nsd[label_name].append(v)

    header = f"{'Label':<35}  {'Dice':>8}" + (f"  {'NSD':>8}" if compute_nsd else "")
    print(header)
    print("-" * len(header))
    all_d, all_n = [], []
    for label_name in LABEL_MAP:
        d_vals = label_dice[label_name]
        mean_d = np.mean(d_vals) if d_vals else float("nan")
        row = f"{label_name:<35}  {mean_d:>8.4f}"
        if compute_nsd:
            n_vals = label_nsd[label_name]
            mean_n = np.mean(n_vals) if n_vals else float("nan")
            row += f"  {mean_n:>8.4f}"
            all_n.extend(n_vals)
        all_d.extend(d_vals)
        print(row)
    print("-" * len(header))
    summary = f"{'mean':<35}  {np.mean(all_d):>8.4f}"
    if compute_nsd:
        summary += f"  {np.mean(all_n):>8.4f}"
    print(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-pattern",   required=True,
                        help="Path template for GT files, e.g. /data/labelsTs/{subject}.nii.gz")
    parser.add_argument("--pred-pattern", required=True,
                        help="Path template for prediction files")
    parser.add_argument("--subjects",     required=True,
                        help="Subject list: a directory (auto-discover), splits JSON, or text file")
    parser.add_argument("--splits-key",   default=None,
                        help="Key to use when --subjects is a splits JSON (e.g. 'test')")
    parser.add_argument("--output",       default="eval_results.json")
    parser.add_argument("--no-nsd",       action="store_true")
    parser.add_argument("--tolerance",    type=float, default=2.0)
    parser.add_argument("--workers",      type=int,   default=8)
    args = parser.parse_args()

    subjects = load_subjects(args.subjects, args.splits_key)
    compute_nsd = not args.no_nsd

    print(f"Evaluating {len(subjects)} subjects | NSD={'on' if compute_nsd else 'off'} | workers={args.workers}")

    all_results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                eval_subject, sub,
                args.gt_pattern, args.pred_pattern,
                compute_nsd, args.tolerance
            ): sub
            for sub in subjects
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            subject, res = future.result()
            all_results[subject] = res

    errors = [s for s, r in all_results.items() if "error" in r]
    print(f"\nDone: {len(all_results) - len(errors)}/{len(subjects)} subjects OK"
          + (f", {len(errors)} errors: {errors}" if errors else ""))

    print_summary(all_results, compute_nsd)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
