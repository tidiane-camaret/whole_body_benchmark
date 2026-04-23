import os
import sys
import json
import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import layers
from hydra import compose, initialize_config_dir

sys.path.insert(0, "/nfs/norasys/notebooks/camaret/repos/patchwork")
import patchwork

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs")
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="config")

img_base  = Path(cfg.paths.nako_dir) / "links"
mask_base = Path(cfg.paths.nako_dir) / "data"
mask_rel  = "30/opportunistic-screening/seg.nii.gz"

splits_path = Path(cfg.paths.results_dir) / "splits.json"
with open(splits_path) as f:
    splits = json.load(f)
subjects_test = splits["test"]

CACHE_DIR    = Path(cfg.paths.results_dir) / "patchwork_cache_raw"
MODEL_PATH   = "/nfs/data/nii/data0/GNC/Analysis/GNC_759/ANALYSIS_whole_body_benchmark/results/patchwork/whole_body_oppscreen_v3"
WEIGHTS_PATH = MODEL_PATH + ".weights/data.0.weights.h5"

nD         = 3
num_labels = 25

LABEL_NAMES = {
     1: "AT_pelvis",            2: "IMAT_pelvis",           3: "Muscle_pelvis",
     4: "AVAT_pelvis",          5: "AT_upper_abdomen",      6: "IMAT_upper_abdomen",
     7: "Muscle_upper_abdomen", 8: "AVAT_upper_abdomen",    9: "AT_thorax",
    10: "IMAT_thorax",         11: "Muscle_thorax",         12: "AVAT_thorax",
    13: "heart",               14: "IVD",                   15: "vertebra_body",
    16: "vertebra_posterior_elements",                       17: "liver",
    18: "pankreas",            19: "aorta",                 20: "kidney_cortex_left",
    21: "kidney_hilus_left",   22: "kidney_medulla_left",   23: "kidney_cortex_right",
    24: "kidney_hilus_right",  25: "kidney_medulla_right",
}

# ── Build model (same architecture as training) ───────────────────────────────
patching = {
    "depth": 4, "ndim": nD,
    "scheme": {"patch_size": [32, 32, 32], "destvox_mm": [2, 2, 2], "fov_mm": [300, 200, 500]},
    "smoothfac_data": 0, "smoothfac_label": 0, "categorial_label": None,
    "interp_type": "NN", "scatter_type": "NN", "normalize_input": "patch_m0s1",
}
cgen = patchwork.CropGenerator(**patching)
themodel = patchwork.PatchWorkModel(
    cgen,
    num_labels=num_labels,
    modelname=MODEL_PATH,
    blockCreator=lambda level, outK, input_shape: patchwork.customLayers.createUnet_v2(
        depth=5, outK=outK, nD=nD, input_shape=input_shape,
        feature_dim=[32, 32, 64, 64, 128],
    ),
    intermediate_out=8,
    intermediate_loss=True,
    finalBlock=layers.Activation("softmax"),
)

# Initialize graph with first available subject
init_subj = next(
    s for s in subjects_test
    if all((CACHE_DIR / f"{s}_ch{ch:04d}.nii.gz").exists() for ch in range(4))
)
init_files = [str(CACHE_DIR / f"{init_subj}_ch{ch:04d}.nii.gz") for ch in range(4)]
print(f"Initializing model with {init_subj}...")
themodel.apply_on_nifti(init_files, num_patches=1, generate_type="random", verbose=False)

print(f"Loading weights from {WEIGHTS_PATH}")
themodel.load_weights(WEIGHTS_PATH)

# ── Eval ──────────────────────────────────────────────────────────────────────
VAL_NUM_PATCHES  = 300
VAL_BG_THRESHOLD = 0.3
VAL_DICE_PATH    = MODEL_PATH + "_eval_dice.txt"
VAL_DICE_JSON    = MODEL_PATH + "_eval_dice.json"


def dice_per_label(pred_prob, gt_int):
    pred_label = np.argmax(pred_prob, axis=-1) + 1
    pred_label[np.max(pred_prob, axis=-1) < VAL_BG_THRESHOLD] = 0
    dices = np.full(num_labels, np.nan, dtype=np.float32)
    for i in range(num_labels):
        cls = i + 1
        p, g = (pred_label == cls), (gt_int == cls)
        union = p.sum() + g.sum()
        if union > 0:
            dices[i] = 2.0 * (p & g).sum() / union
    return dices


per_subject_dices = {}
for subj in subjects_test:
    fnames = [str(CACHE_DIR / f"{subj}_ch{ch:04d}.nii.gz") for ch in range(4)]
    mask_path = str(mask_base / subj / mask_rel)
    if not all(os.path.exists(f) for f in fnames) or not os.path.exists(mask_path):
        print(f"  {subj}: missing files, skipping")
        continue

    print(f"  {subj}", end="", flush=True)
    res = themodel.apply_on_nifti(
        fnames,
        num_patches=VAL_NUM_PATCHES,
        generate_type="random",
        verbose=0,
    )
    prob = np.array(res, dtype=np.float32)

    gt_nii  = nib.load(mask_path)
    gt_data = np.array(gt_nii.get_fdata(), dtype=np.int16)

    dices = dice_per_label(prob, gt_data)
    per_subject_dices[subj] = dices
    print(f"  mean Dice = {float(np.nanmean(dices)):.4f}")

print(f"\nDone. {len(per_subject_dices)} subjects evaluated.")

all_dices    = np.stack(list(per_subject_dices.values()), axis=0)
label_mean   = np.nanmean(all_dices, axis=0)
label_std    = np.nanstd(all_dices,  axis=0)
overall_mean = float(np.nanmean(label_mean))

lines = ["=" * 72,
         "Patchwork Eval — Dice per label (oppscreen MRI)",
         f"Model  : {MODEL_PATH}",
         f"Date   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
         f"Subjects: {len(per_subject_dices)}",
         f"Patches: {VAL_NUM_PATCHES}, bg_threshold={VAL_BG_THRESHOLD}",
         "=" * 72, ""]
col_w = max(len(n) for n in LABEL_NAMES.values()) + 2
for i in range(num_labels):
    cls  = i + 1
    name = LABEL_NAMES.get(cls, f"class_{cls}")
    m, s = label_mean[i], label_std[i]
    tag  = f"{m:.4f} ± {s:.4f}" if not np.isnan(m) else "n/a"
    lines.append(f"  {cls:>3d}  {name:<{col_w}s}  {tag}")
lines += ["", f"Overall mean Dice: {overall_mean:.4f}", "=" * 72]

with open(VAL_DICE_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines[-5:]))
print(f"\nResults written to:\n  {VAL_DICE_PATH}")

with open(VAL_DICE_JSON, "w") as f:
    json.dump({
        "model": MODEL_PATH, "date": datetime.datetime.now().isoformat(),
        "num_subjects": len(per_subject_dices),
        "overall_mean_dice": overall_mean,
        "per_label": {
            LABEL_NAMES.get(i+1, f"class_{i+1}"): {
                "mean": float(label_mean[i]) if not np.isnan(label_mean[i]) else None,
                "std":  float(label_std[i])  if not np.isnan(label_std[i])  else None,
            } for i in range(num_labels)
        },
        "per_subject": {
            subj: {LABEL_NAMES.get(i+1, f"class_{i+1}"): (float(d[i]) if not np.isnan(d[i]) else None)
                   for i in range(num_labels)}
            for subj, d in per_subject_dices.items()
        },
    }, f, indent=2)
print(f"JSON written to:\n  {VAL_DICE_JSON}")
