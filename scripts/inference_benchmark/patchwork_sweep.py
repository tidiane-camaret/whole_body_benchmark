"""
Patchwork inference parameter sweep.

Runs inference across configurations that vary patch budget, budget distribution,
and lazyEval pruning. Measures wall-clock time and Dice against GT.

Results are saved incrementally to results/inference_benchmark/patchwork_sweep.json.
Completed configs are skipped on re-run.

Usage (from project root):
    python scripts/inference_benchmark/patchwork_sweep.py
    python scripts/inference_benchmark/patchwork_sweep.py --dry-run
    python scripts/inference_benchmark/patchwork_sweep.py --configs budget_5120 lazy_0.7
    python scripts/inference_benchmark/patchwork_sweep.py --subjects 100003 100005
"""

import sys
import json
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from timeit import default_timer as timer
from scipy.ndimage import zoom

sys.path.append("/software/")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = (
    "/nfs/data/nii/data1/Analysis/camaret___whole_body_benchmark"
    "/ANALYSIS_ana001/patchwork_oppscreen_all/model_patchwork.json"
)
BENCH_ROOT = Path(
    "/nfs/data/nii/data1/Analysis/camaret___whole_body_benchmark"
    "/ANALYSIS_ana001/patchwork_inference_benchmark"
)
IMG_PATTERN  = str(BENCH_ROOT / "imagesTs/{subject}.nii")
GT_PATTERN   = str(BENCH_ROOT / "labelsTs/{subject}.nii.gz")
PRED_SWEEP   = BENCH_ROOT / "predsTs_sweep"
OUTPUT       = Path("results/inference_benchmark/patchwork_sweep.json")

ALL_SUBJECTS = [
    "100003", "100006", "100009", "100020", "100027",
    "100005", "100008", "100015", "100024", "100029",
]

# QMap weights — identical to patchwork.py baseline
_W = [0.0] * 86
_W[0]  = -1.5
_W[71] =  1.5
_W[58] =  1.5

# Fixed params shared by every config
FIXED = dict(
    generate_type="random",
    augment={},
    scale_to_original=True,
    out_typ="atls",
    QMapply_paras={"weights": _W},
    level="mix",
)

# ---------------------------------------------------------------------------
# Experiment grid
#
# Three axes:
#   1. Total patch budget  — vary branch_factor[-1] (linear in fine patches)
#   2. Budget distribution — same ∏(branch_factor)=128, different shape
#   3. lazyEval pruning    — attention-guided patch skipping at fixed budget
#
# Fine patches per run = num_chunks × repetitions × ∏(branch_factor)
# ---------------------------------------------------------------------------
CONFIGS = {
    # --- Axis 1: total patch budget ---
    # branch_factor=[4,4,k] → fine patches = 20 × 4 × 16k
    "budget_2560":  dict(repetitions=4, num_chunks=20, branch_factor=[4, 4,  2]),  #  2,560
    "budget_5120":  dict(repetitions=4, num_chunks=20, branch_factor=[4, 4,  4]),  #  5,120
    "baseline":     dict(repetitions=4, num_chunks=20, branch_factor=[4, 4,  8]),  # 10,240
    "budget_20480": dict(repetitions=4, num_chunks=20, branch_factor=[4, 4, 16]),  # 20,480

    # --- Axis 2: distribution across levels (∏ branch_factor = 128 each) ---
    # Same total fine patches as baseline, different hierarchical weighting
    "dist_top_heavy": dict(repetitions=4, num_chunks=20, branch_factor=[8, 4, 4]),  # coarse-heavy
    "dist_mid_heavy": dict(repetitions=4, num_chunks=20, branch_factor=[4, 8, 4]),  # mid-heavy
    # "baseline" already represents bottom-heavy [4,4,8]

    # --- Axis 3: lazyEval pruning at baseline budget ---
    # Fraction = proportion of patches forwarded to each deeper level
    "lazy_0.9": dict(repetitions=4, num_chunks=20, branch_factor=[4, 4, 8], lazyEval=0.9),
    "lazy_0.7": dict(repetitions=4, num_chunks=20, branch_factor=[4, 4, 8], lazyEval=0.7),
    "lazy_0.5": dict(repetitions=4, num_chunks=20, branch_factor=[4, 4, 8], lazyEval=0.5),
}

# ---------------------------------------------------------------------------
# Label map (matches eval.py)
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "AT_pelvis": 1, "IMAT_pelvis": 2, "Muscle_pelvis": 3, "AVAT_pelvis": 4,
    "AT_upper_abdomen": 5, "IMAT_upper_abdomen": 6, "Muscle_upper_abdomen": 7,
    "AVAT_upper_abdomen": 8, "AT_thorax": 9, "IMAT_thorax": 10,
    "Muscle_thorax": 11, "AVAT_thorax": 12, "heart": 13, "IVD": 14,
    "vertebra_body": 15, "vertebra_posterior_elements": 16, "liver": 17,
    "pankreas": 18, "aorta": 19, "kidney_cortex_left": 20,
    "kidney_hilus_left": 21, "kidney_medulla_left": 22,
    "kidney_cortex_right": 23, "kidney_hilus_right": 24,
    "kidney_medulla_right": 25,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return float(2 * np.logical_and(a, b).sum() / denom)


def evaluate(pred_path: str, gt_path: str) -> dict:
    gt_nii   = nib.load(gt_path)
    pred_nii = nib.load(pred_path)
    gt   = np.asarray(gt_nii.dataobj, dtype=np.int32)
    pred = np.asarray(pred_nii.dataobj, dtype=np.int32)
    if pred.ndim == 4:
        pred = pred[..., 0]
    if pred.shape != gt.shape:
        factors = tuple(g / p for g, p in zip(gt.shape, pred.shape))
        pred = zoom(pred, factors, order=0)
    return {
        name: _dice(gt == k, pred == k)
        for name, k in LABEL_MAP.items()
    }


def _patch_count(cfg: dict) -> int:
    bf = cfg.get("branch_factor", 1)
    product = bf if isinstance(bf, int) else int(np.prod(bf))
    return cfg.get("num_chunks", 1) * cfg.get("repetitions", 1) * product


def _summarise(subjects: dict) -> dict:
    times = [v["time_s"] for v in subjects.values() if "time_s" in v]
    all_dice = [
        d for v in subjects.values()
        for d in v.get("dice", {}).values()
        if not (isinstance(d, float) and np.isnan(d))
    ]
    return {
        "mean_time_s": float(np.mean(times)) if times else None,
        "total_time_s": float(np.sum(times)) if times else None,
        "mean_dice": float(np.mean(all_dice)) if all_dice else None,
    }


def load_results(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(results: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def print_table(results: dict):
    print(f"\n{'Config':<20}  {'Fine patches':>12}  {'Mean time(s)':>12}  {'Mean Dice':>10}")
    print("-" * 62)
    for name, entry in results.items():
        params = entry.get("params", {})
        n = _patch_count(params)
        s = entry.get("summary", {})
        t = f"{s['mean_time_s']:.1f}" if s.get("mean_time_s") else "—"
        d = f"{s['mean_dice']:.4f}"  if s.get("mean_dice")   else "—"
        print(f"{name:<20}  {n:>12,}  {t:>12}  {d:>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs",  nargs="+", default=None,
                        help="Run only these config names (default: all)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Run only these subjects (default: all 10)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print config table and exit without running")
    parser.add_argument("--output",   default=str(OUTPUT))
    args = parser.parse_args()

    subjects    = args.subjects or ALL_SUBJECTS
    output_path = Path(args.output)
    configs     = {k: CONFIGS[k] for k in (args.configs or CONFIGS)}

    # Dry-run: show what would run
    if args.dry_run:
        print(f"{'Config':<20}  {'Fine patches':>12}  {'Params'}")
        print("-" * 70)
        for name, cfg in configs.items():
            print(f"{name:<20}  {_patch_count(cfg):>12,}  {cfg}")
        return

    print(f"Configs  : {list(configs)}")
    print(f"Subjects : {subjects}")
    print(f"Output   : {output_path}")

    results = load_results(output_path)

    # Determine which configs still need running
    todo = {
        name: cfg for name, cfg in configs.items()
        if name not in results or set(subjects) - set(results[name].get("subjects", {}))
    }
    if not todo:
        print("All configs already complete.")
        print_table(results)
        return

    import patchwork2.model as patchwork

    print(f"\nLoading model …")
    model = patchwork.PatchWorkModel.load(MODEL_PATH, immediate_init=True)
    print("Model loaded.\n")

    for config_name, cfg in todo.items():
        fine_patches = _patch_count(cfg)
        print(f"{'='*60}")
        print(f"Config: {config_name}  ({fine_patches:,} fine patches)")
        print(f"Params: {cfg}")
        print(f"{'='*60}")

        pred_dir = PRED_SWEEP / config_name
        pred_dir.mkdir(parents=True, exist_ok=True)

        entry = results.get(config_name, {"params": {**FIXED, **cfg}, "subjects": {}})

        for subject in subjects:
            if subject in entry["subjects"]:
                print(f"  {subject}: already done, skipping.")
                continue

            img_path  = IMG_PATTERN.format(subject=subject)
            pred_path = str(pred_dir / f"{subject}.nii.gz")
            gt_path   = GT_PATTERN.format(subject=subject)

            print(f"  {subject}: running …", flush=True)
            t0 = timer()
            try:
                model.apply_on_nifti(img_path, pred_path, **FIXED, **cfg)
                elapsed = timer() - t0
                dice_scores = evaluate(pred_path, gt_path)
                mean_d = float(np.nanmean(list(dice_scores.values())))
                print(f"  {subject}: {elapsed:.1f}s  mean Dice={mean_d:.4f}")
                entry["subjects"][subject] = {"time_s": elapsed, "dice": dice_scores}
            except Exception as e:
                elapsed = timer() - t0
                print(f"  {subject}: ERROR after {elapsed:.1f}s — {e}")
                entry["subjects"][subject] = {"error": str(e)}

            # save after every subject so progress is not lost
            entry["summary"] = _summarise(entry["subjects"])
            results[config_name] = entry
            save_results(results, output_path)

        print(f"  Summary: {entry['summary']}\n")

    print_table(results)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
