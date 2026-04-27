#!/usr/bin/env python3
"""
Evaluation of the oppscreen patchwork 3D model on validation subjects.

Outputs (in METRICS_DIR):
  predictions/            one pred.nii.gz per subject
  metrics_per_subject.csv label × subject Dice and NSD
  metrics_per_label.csv   mean / std / median across subjects per label
  timing.csv              per-subject inference time + aggregate
  flops.json              FLOPs per CNN block and total estimate
"""

import sys, os, json, time, gc
from pathlib import Path
from glob import glob

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_erosion

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, "/software")
import matplotlib; matplotlib.use("Agg")
import tensorflow as tf
from patchwork2.customLayers import *
from patchwork2.improc_utils import *
import patchwork2.model as patchwork

# ─── CONFIG ────────────────────────────────────────────────────────────────────

config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs")
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="config")

img_base  = Path(cfg.paths.nako_dir) / "links"
mask_base = Path(cfg.paths.nako_dir) / "data"
img_glob  = "30/3D_GRE_TRA_4/3D_GRE_TRA_W_COMPOSE*_s*.nii"
mask_rel  = "30/opportunistic-screening/seg.nii.gz"

splits_path = Path(cfg.paths.data_dir) / "splits_966.json"
with open(splits_path) as f:
    splits = json.load(f)
subjects_val = splits["test"]

#MODEL_DIR   = Path(cfg.paths.results_dir) / "patchwork" / "subsetFW"
MODEL_DIR = Path("/nfs/data/nii/data0/GNC/Analysis/GNC_759/ANALYSIS_wholebody/whole_body_benchmark")
PRED_DIR    = Path(cfg.paths.results_dir) / "patchwork" / "predictions"
METRICS_DIR = Path(cfg.paths.results_dir) / "patchwork" / "metrics"
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 25    # labels 1..25
NSD_TAU_MM  = 2.0   # surface-distance tolerance

# ─── APPLY PARAMETERS (mirror of deploy job) ───────────────────────────────────

qm_weights       = [0.0] * 86
qm_weights[0]    = -1.5
qm_weights[71]   =  1.5
qm_weights[58]   =  1.5

APPLY_PARAMS = dict(
    generate_type     = "random",
    augment           = {},
    crop_fdim         = [2, 3],
    repetitions       = 4,
    num_chunks        = 20,
    scale_to_original = False,
    branch_factor     = [4, 4, 8],
    out_typ           = "atls",
    level             = "mix",
    QMapply_paras     = {"weights": qm_weights},
)

# ─── METRICS ───────────────────────────────────────────────────────────────────

def dice_score(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return float(2 * inter / denom) if denom > 0 else float("nan")


def nsd_score(pred_bin, gt_bin, spacing, tau=NSD_TAU_MM):
    """Normalised Surface Distance at tolerance `tau` mm (symmetric)."""
    pred_surf = pred_bin ^ binary_erosion(pred_bin)
    gt_surf   = gt_bin   ^ binary_erosion(gt_bin)
    if not pred_surf.any() or not gt_surf.any():
        return float("nan")
    pred_dist = distance_transform_edt(~pred_surf, sampling=spacing)
    gt_dist   = distance_transform_edt(~gt_surf,   sampling=spacing)
    num   = (pred_dist[gt_surf] <= tau).sum() + (gt_dist[pred_surf] <= tau).sum()
    denom = gt_surf.sum() + pred_surf.sum()
    return float(num / denom)


def metrics_for_subject(pred_arr, gt_arr, spacing):
    rows = []
    for label in range(1, NUM_CLASSES + 1):
        pb = pred_arr == label
        gb = gt_arr   == label
        rows.append({
            "label": label,
            "dice":  dice_score(pb, gb),
            "nsd":   nsd_score(pb, gb, spacing),
            "pred_voxels": int(pb.sum()),
            "gt_voxels":   int(gb.sum()),
        })
    return rows

# ─── FLOPS ESTIMATION ──────────────────────────────────────────────────────────

def block_flops(keras_model, input_shape):
    """
    FLOPs for a single forward pass of one CNN block.
    `input_shape` = spatial dims + channels, e.g. (32, 32, 32, 2).
    Returns int (total float ops) or None on failure.
    """
    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2,
        )

        @tf.function
        def fwd(x):
            return keras_model(x, training=False)

        spec     = tf.TensorSpec([1, *input_shape], tf.float32)
        concrete = fwd.get_concrete_function(spec)
        frozen   = convert_variables_to_constants_v2(concrete)
        graph    = frozen.graph
        with graph.as_default():
            opts  = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            prof  = tf.compat.v1.profiler.profile(
                graph=graph,
                run_meta=tf.compat.v1.RunMetadata(),
                cmd="op",
                options=opts,
            )
            return int(prof.total_float_ops)
    except Exception as e:
        print(f"  FLOPs estimation failed: {e}")
        return None


def estimate_total_flops(themodel, branch_factor, repetitions):
    """
    Estimate total FLOPs for one apply call.
    Each level l processes branch_factor[l] patches per coarser patch.
    Total patches at level l = prod(branch_factor[:l]).
    """
    depth  = len(themodel.blocks)
    bf     = branch_factor if isinstance(branch_factor, list) else [branch_factor] * depth
    # pad to depth if needed
    while len(bf) < depth:
        bf.append(bf[-1])

    patch_size = [32, 32, 32]          # as configured
    n_channels = 2                     # after crop_fdim=[2,3]

    total_flops = 0
    patches_at_level = 1
    block_flops_list = []
    for lvl, block in enumerate(themodel.blocks):
        n_patches = patches_at_level * repetitions
        f = block_flops(block, patch_size + [n_channels])
        block_flops_list.append({"level": lvl, "block_flops": f, "n_patches": n_patches})
        if f is not None:
            total_flops += f * n_patches
        if lvl < len(bf):
            patches_at_level *= bf[lvl]

    return total_flops if total_flops > 0 else None, block_flops_list

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

modelfi = str(MODEL_DIR / "model_patchwork")
print(f"\n>>> Loading model: {modelfi}")
themodel = patchwork.PatchWorkModel.load(modelfi, immediate_init=True, notmpfile=True)
themodel.summary()

print("\n>>> Estimating FLOPs ...")
total_flops, block_flops_info = estimate_total_flops(
    themodel,
    branch_factor=APPLY_PARAMS["branch_factor"],
    repetitions=APPLY_PARAMS["repetitions"],
)

flops_report = {
    "blocks":        block_flops_info,
    "total_estimate": total_flops,
    "total_gflops":  round(total_flops / 1e9, 3) if total_flops else None,
    "note": (
        "Total = sum over levels of (block_flops × n_patches × repetitions). "
        "n_patches grows by branch_factor at each level."
    ),
}
with open(METRICS_DIR / "flops.json", "w") as f:
    json.dump(flops_report, f, indent=2)
print(json.dumps(flops_report, indent=2))

# ─── INFERENCE LOOP ────────────────────────────────────────────────────────────

all_records  = []
timing_rows  = []
skipped      = []

for sid in subjects_val:
    img_matches = sorted(glob(str(img_base / sid / img_glob)))
    mask_path   = mask_base / sid / mask_rel

    if not img_matches:
        print(f"[SKIP] {sid}: no image match")
        skipped.append({"subject": sid, "reason": "image missing"})
        continue
    if not mask_path.exists():
        print(f"[SKIP] {sid}: mask missing")
        skipped.append({"subject": sid, "reason": "mask missing"})
        continue

    img_path  = img_matches[0]
    pred_path = str(PRED_DIR / f"{sid}_pred.nii.gz")

    print(f"\n>>> {sid}  [{img_path}]")
    t0 = time.perf_counter()
    themodel.apply_on_nifti(img_path, pred_path, **APPLY_PARAMS)
    t_infer = time.perf_counter() - t0
    print(f"    done in {t_infer:.1f}s")

    gt_nib   = nib.load(str(mask_path))
    pred_nib = nib.load(pred_path)

    gt_arr   = np.asarray(gt_nib.dataobj,   dtype=np.int16)
    pred_arr = np.asarray(pred_nib.dataobj, dtype=np.int16)
    spacing  = np.abs(np.diag(pred_nib.affine)[:3])   # mm/voxel from affine

    rows = metrics_for_subject(pred_arr, gt_arr, spacing)
    for r in rows:
        r["subject"]   = sid
        r["t_infer_s"] = round(t_infer, 2)
    all_records.extend(rows)
    timing_rows.append({"subject": sid, "t_infer_s": round(t_infer, 2)})

    gc.collect()

# ─── SAVE & SUMMARISE ──────────────────────────────────────────────────────────

df = pd.DataFrame(all_records)
df.to_csv(METRICS_DIR / "metrics_per_subject.csv", index=False)

# per-label summary across subjects
summary = (
    df.groupby("label")[["dice", "nsd"]]
    .agg(["mean", "std", "median", "min", "max"])
    .round(4)
)
summary.columns = ["_".join(c) for c in summary.columns]
summary.to_csv(METRICS_DIR / "metrics_per_label.csv")

# timing summary
t_df = pd.DataFrame(timing_rows)
t_df.to_csv(METRICS_DIR / "timing.csv", index=False)

if skipped:
    pd.DataFrame(skipped).to_csv(METRICS_DIR / "skipped.csv", index=False)

# ─── PRINT ─────────────────────────────────────────────────────────────────────

print("\n\n=== Per-label Dice ===")
print(summary[["dice_mean", "dice_std", "dice_median"]].to_string())

print(f"\n\n=== Per-label NSD @{NSD_TAU_MM}mm ===")
print(summary[["nsd_mean", "nsd_std", "nsd_median"]].to_string())

print("\n\n=== Inference timing ===")
print(f"  n subjects  : {len(timing_rows)}")
print(f"  mean        : {t_df['t_infer_s'].mean():.1f}s")
print(f"  median      : {t_df['t_infer_s'].median():.1f}s")
print(f"  min / max   : {t_df['t_infer_s'].min():.1f}s / {t_df['t_infer_s'].max():.1f}s")

if total_flops:
    print(f"\n=== FLOPs (estimated) ===")
    print(f"  Total per inference : {total_flops/1e9:.2f} GFLOPs")
    for b in block_flops_info:
        gf = f"{b['block_flops']/1e9:.3f}" if b["block_flops"] else "N/A"
        print(f"  Level {b['level']}: {gf} GFLOPs/patch  ×  {b['n_patches']} patches")

print(f"\nAll results in: {METRICS_DIR}")
