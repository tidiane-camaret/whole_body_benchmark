import os
import sys
import glob
import gc
import json
import random
import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, "/nfs/norasys/notebooks/camaret/repos/patchwork")
import patchwork

# ── GPU ──────────────────────────────────────────────────────────────────────
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# ── Config ───────────────────────────────────────────────────────────────────
config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs")
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="config")
print(OmegaConf.to_yaml(cfg))

img_base = Path(cfg.paths.nako_dir) / "links"
mask_base = Path(cfg.paths.nako_dir) / "data"
img_glob  = "30/3D_GRE_TRA_4/3D_GRE_TRA_W_COMPOSE*_s*.nii"
mask_rel  = "30/opportunistic-screening/seg.nii.gz"

# ── Splits ───────────────────────────────────────────────────────────────────
splits_path = Path(cfg.paths.results_dir) / "splits.json"
with open(splits_path) as f:
    splits = json.load(f)
subjects_train, subjects_test = splits["train"], splits["test"]

# ── Channel cache (raw, no normalisation) ────────────────────────────────────
CACHE_DIR = Path(cfg.paths.results_dir) / "patchwork_cache_raw"
CACHE_DIR.mkdir(exist_ok=True)


def get_channel_path(subject, ch, img_base, img_glob, cache_dir):
    dst = cache_dir / f"{subject}_ch{ch:04d}.nii.gz"
    if dst.exists():
        return str(dst)
    files = glob.glob(str(img_base / subject / img_glob))
    if not files:
        return None
    img    = nib.load(files[0])
    data   = img.get_fdata(dtype=np.float32)
    ch_data = data[..., ch] if data.ndim == 4 else data
    out = nib.Nifti1Image(ch_data, img.affine, img.header)
    out.header.set_data_shape(ch_data.shape)
    nib.save(out, dst)
    return str(dst)


all_subjects_used = subjects_train + subjects_test
contrasts   = [{} for _ in range(4)]
labels_dict = {}

print("Preparing file references (splitting channels on first run)...")
for subject in all_subjects_used:
    for ch in range(4):
        path = get_channel_path(subject, ch, img_base, img_glob, CACHE_DIR)
        if path:
            contrasts[ch][subject] = path
    mask_path = str(mask_base / subject / mask_rel)
    if os.path.exists(mask_path):
        labels_dict[subject] = mask_path

valid = [
    s for s in all_subjects_used
    if all(s in contrasts[ch] for ch in range(4)) and s in labels_dict
]
print(f"{len(valid)} subjects ready")

contrasts   = [{s: d[s] for s in valid if s in d} for d in contrasts]
labels_pw   = [{s: labels_dict[s] for s in valid}]
subjects_pw = valid
valid_ids   = list(range(len(subjects_train), len(valid)))

# ── Experiment config ─────────────────────────────────────────────────────────
nD = 3

MODEL_PATH = "/nfs/data/nii/data0/GNC/Analysis/GNC_759/ANALYSIS_whole_body_benchmark/results/patchwork/whole_body_oppscreen_v3"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

patching = {
    "depth": 4,
    "ndim": nD,
    "scheme": {
        "patch_size": [32, 32, 32],
        "destvox_mm": [2, 2, 2],
        "fov_mm": [300, 200, 500],
    },
    "smoothfac_data": 0,
    "smoothfac_label": 0,
    "categorial_label": None,
    "interp_type": "NN",
    "scatter_type": "NN",
    "normalize_input": "patch_m0s1",
}

loading = {
    "nD": nD,
    "align_physical": False,
    "crop_fdim": [2, 3],
    "crop_fdim_labels": None,
    "crop_only_nonzero": False,
    "threshold": 0.5,
    "add_inverted_label": False,
    "one_hot_index_list": list(range(1, 26)),
    "exclude_incomplete_labels": 1,
    "annotations_selector": None,
}

training = {
    "num_patches": 32,
    "epochs": 4,
    "num_its": 100,
    "num_patches_to_train": 3200,
    "balance": {"ratio": 0.8},
    "fit_type": "custom",
    "augment": {"dphi": 0.3, "flip": [0, 0, 0], "dscale": [0.3, 0.3, 0.3], "vscale": 0.5},
}

outer_its      = 100
# NOTE: validation tag "mjvalset" — filter subjects if a tag file is available.
samples_per_it = 10

# ── Data loader ───────────────────────────────────────────────────────────────
def get_data(subjects_subset=None, n=None):
    subs = subjects_subset if subjects_subset else subjects_pw
    ctrs = [{s: d[s] for s in subs if s in d} for d in contrasts]
    lbls = [{s: labels_pw[0][s] for s in subs if s in labels_pw[0]}]
    return patchwork.improc_utils.load_data_structured(
        contrasts=ctrs, labels=lbls, subjects=subs, max_num_data=n, **loading
    )


print("Loading one subject to initialise model...")
with tf.device("/cpu:0"):
    tset, lset, rset, subjs = get_data(n=1)

print(f"Image shape: {tset[0].shape}   Label shape: {lset[0].shape}")
num_labels = lset[0].shape[nD + 1]
print(f"num_labels = {num_labels}")

# ── Model ─────────────────────────────────────────────────────────────────────
reinit_model = True

if os.path.isfile(MODEL_PATH + ".json") and not reinit_model:
    print("Loading existing model...")
    themodel = patchwork.PatchWorkModel.load(MODEL_PATH, immediate_init=True, notmpfile=True,
                                             custom_objects={"Activation": layers.Activation})
else:
    print("Creating new model...")
    network = {
        "num_labels": num_labels,
        "modelname": MODEL_PATH,
        "blockCreator": lambda level, outK, input_shape:
            patchwork.customLayers.createUnet_v2(
                depth=5, outK=outK, nD=nD,
                input_shape=input_shape,
                feature_dim=[32, 32, 64, 64, 128],
            ),
        "intermediate_out": 8,
        "intermediate_loss": True,
        "finalBlock": layers.Activation("softmax"),
    }

    cgen     = patchwork.CropGenerator(**patching)
    themodel = patchwork.PatchWorkModel(cgen, **network)

    print("Initialising weights...")
    themodel.apply_full(
        tset[0][0:1, ...], resolution=rset[0],
        repetitions=10, generate_type="random", verbose=True,
    )

themodel.summary()

# ── Training loop ─────────────────────────────────────────────────────────────
train_subjects = [s for s in subjects_pw if s not in [subjects_pw[i] for i in valid_ids]]
val_subjects   = [subjects_pw[i] for i in valid_ids]

for i in range(outer_its):
    print(f"\n{'='*60}\nOuter iteration {i+1}/{outer_its}")

    subset = random.sample(train_subjects, min(samples_per_it, len(train_subjects)))

    with tf.device("/cpu:0"):
        tset, lset, rset, subjs = get_data(subjects_subset=subset + val_subjects)

    if len(tset) == 0:
        print("No data loaded, skipping")
        continue

    val_ids_iter = [j for j, s in enumerate(subjs) if s in set(val_subjects)]

    themodel.train(
        tset, lset, resolutions=rset,
        **training,
        loss=tf.keras.losses.categorical_crossentropy,
        batch_size=1,
        valid_ids=val_ids_iter,
        autosave=True,
        verbose=2,
        inc_train_cycle=False,
        debug=False,
        patch_on_cpu=True,
        parallel=False,
    )

    with tf.device("/cpu:0"):
        del tset, lset
        gc.collect()

print("\nTraining complete.")

# ── Quick single-subject inference ────────────────────────────────────────────
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

val_subject = subjects_pw[valid_ids[0]]
print(f"Running inference on: {val_subject}")

with tf.device("/cpu:0"):
    tset_val, lset_val, rset_val, _ = get_data(subjects_subset=[val_subject])

pred = themodel.apply_full(
    tset_val[0][0:1, ...],
    resolution=rset_val[0],
    num_patches=200,
    generate_type="random",
    verbose=2,
)
pred = np.array(pred)
img  = np.array(tset_val[0])
print(f"pred shape: {pred.shape}")

pred_label = np.argmax(pred, axis=-1) + 1
pred_label[np.max(pred, axis=-1) < 0.3] = 0

gt_onehot = np.array(lset_val[0][0])
gt_label  = np.argmax(gt_onehot, axis=-1) + 1
gt_label[np.max(gt_onehot, axis=-1) == 0] = 0

dices = np.full(num_labels, np.nan)
for i in range(num_labels):
    cls   = i + 1
    p     = (pred_label == cls)
    g     = (gt_label   == cls)
    union = p.sum() + g.sum()
    if union > 0:
        dices[i] = 2.0 * (p & g).sum() / union

mean_dice = float(np.nanmean(dices))
print(f"\nMean Dice ({np.sum(~np.isnan(dices))}/{num_labels} present classes): {mean_dice:.4f}")

mid_slice = pred_label.shape[2] // 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img[0, :, :, mid_slice, 0].T, cmap="gray", origin="lower")
axes[0].set_title("Input ch2 (fat, raw)")
axes[1].imshow(gt_label[:, :, mid_slice].T, origin="lower", interpolation="nearest")
axes[1].set_title("Ground truth")
axes[2].imshow(pred_label[:, :, mid_slice].T, origin="lower", interpolation="nearest")
axes[2].set_title("Prediction")
for ax in axes:
    ax.axis("off")
plt.suptitle(f"Subject: {val_subject}  |  mid axial slice z={mid_slice}", y=1.01)
plt.tight_layout()
fig.savefig(MODEL_PATH + "_quickval.png", dpi=150, bbox_inches="tight")
plt.close(fig)

valid_mask  = ~np.isnan(dices)
valid_idx   = np.where(valid_mask)[0]
valid_dices = dices[valid_mask]
valid_names = [f"{i+1} {LABEL_NAMES.get(i+1, f'cls_{i+1}')}" for i in valid_idx]
order       = np.argsort(valid_dices)
sorted_d    = valid_dices[order]
sorted_n    = [valid_names[k] for k in order]
colors      = ["#d62728" if d < 0.5 else "#ff7f0e" if d < 0.7 else "#2ca02c" for d in sorted_d]

fig, ax = plt.subplots(figsize=(8, max(6, len(sorted_d) * 0.3)))
ax.barh(range(len(sorted_d)), sorted_d, color=colors, height=0.8)
ax.set_yticks(range(len(sorted_d)))
ax.set_yticklabels(sorted_n, fontsize=9)
ax.set_xlabel("Dice coefficient")
ax.set_xlim(0, 1)
ax.axvline(mean_dice, color="black", linestyle="--", linewidth=1.2, label=f"mean = {mean_dice:.3f}")
ax.axvline(0.5, color="#d62728", linestyle=":", linewidth=0.8)
ax.axvline(0.7, color="#ff7f0e", linestyle=":", linewidth=0.8)
ax.legend(loc="lower right", fontsize=8)
ax.set_title(f"Per-label Dice — {val_subject}")
plt.tight_layout()
fig.savefig(MODEL_PATH + "_quickval_dice.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Full validation ───────────────────────────────────────────────────────────
def dice_per_label(pred_prob, gt_mask, num_labels=25, bg_threshold=0.3):
    pred_label = np.argmax(pred_prob, axis=-1) + 1
    pred_label[np.max(pred_prob, axis=-1) < bg_threshold] = 0
    dices = np.full(num_labels, np.nan, dtype=np.float32)
    for i in range(num_labels):
        cls   = i + 1
        p     = (pred_label == cls)
        g     = (gt_mask    == cls)
        union = p.sum() + g.sum()
        if union == 0:
            continue
        dices[i] = 2.0 * (p & g).sum() / union
    return dices


VAL_DICE_PATH    = MODEL_PATH + "_val_dice.txt"
VAL_DICE_JSON    = MODEL_PATH + "_val_dice.json"
VAL_NUM_PATCHES  = 300
VAL_BG_THRESHOLD = 0.3

val_subjects_list = [subjects_pw[i] for i in valid_ids]
print(f"Evaluating {len(val_subjects_list)} validation subjects → {VAL_DICE_PATH}")

per_subject_dices = {}
for subj_idx, subj in enumerate(val_subjects_list):
    print(f"  [{subj_idx+1}/{len(val_subjects_list)}] {subj}", end="", flush=True)

    with tf.device("/cpu:0"):
        tset_v, lset_v, rset_v, _ = get_data(subjects_subset=[subj])

    if len(tset_v) == 0:
        print(" — skipped (no data)")
        continue

    pred_prob = themodel.apply_full(
        tset_v[0][0:1, ...],
        resolution=rset_v[0],
        num_patches=VAL_NUM_PATCHES,
        generate_type="random",
        verbose=0,
    )
    pred_prob_vol = pred_prob[0] if isinstance(pred_prob, list) else pred_prob
    pred_prob_vol = np.array(pred_prob_vol, dtype=np.float32)

    gt_onehot = lset_v[0][0]
    gt_int    = np.argmax(gt_onehot, axis=-1) + 1
    gt_int[np.max(gt_onehot, axis=-1) == 0] = 0

    per_subject_dices[subj] = dice_per_label(
        pred_prob_vol, gt_int,
        num_labels=num_labels,
        bg_threshold=VAL_BG_THRESHOLD,
    )
    print(f"  mean Dice = {float(np.nanmean(per_subject_dices[subj])):.4f}")

    with tf.device("/cpu:0"):
        del tset_v, lset_v, pred_prob_vol, gt_onehot
        gc.collect()

print(f"\nDone. {len(per_subject_dices)} subjects evaluated.")

all_dices    = np.stack(list(per_subject_dices.values()), axis=0)
label_mean   = np.nanmean(all_dices, axis=0)
label_std    = np.nanstd(all_dices,  axis=0)
overall_mean = float(np.nanmean(label_mean))

lines = ["=" * 72,
         "Patchwork Validation — Dice per label (oppscreen MRI)",
         f"Model     : {MODEL_PATH}",
         f"Date      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
         f"Val set   : {len(per_subject_dices)} subjects",
         f"Patches   : {VAL_NUM_PATCHES} per subject, bg_threshold={VAL_BG_THRESHOLD}",
         "=" * 72, "",
         "── Per-subject mean Dice " + "-" * 48]
for subj, dices in per_subject_dices.items():
    lines.append(f"  {subj:<20s}  mean = {np.nanmean(dices):.4f}")
lines += ["", "── Per-label summary (mean ± std across subjects) " + "-" * 22]
col_w = max(len(n) for n in LABEL_NAMES.values()) + 2
for i in range(num_labels):
    cls  = i + 1
    name = LABEL_NAMES.get(cls, f"class_{cls}")
    m, s = label_mean[i], label_std[i]
    if np.isnan(m):
        lines.append(f"  {cls:>3d}  {name:<{col_w}s}  n/a")
    else:
        lines.append(f"  {cls:>3d}  {name:<{col_w}s}  {m:.4f} ± {s:.4f}")
lines += ["", f"Overall mean Dice (macro, NaN-safe): {overall_mean:.4f}", "=" * 72]

with open(VAL_DICE_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines[-8:]))
print(f"\nFull results written to:\n  {VAL_DICE_PATH}")

results_json = {
    "model": MODEL_PATH,
    "date": datetime.datetime.now().isoformat(),
    "num_val_subjects": len(per_subject_dices),
    "bg_threshold": VAL_BG_THRESHOLD,
    "overall_mean_dice": overall_mean,
    "per_label": {
        LABEL_NAMES.get(i+1, f"class_{i+1}"): {
            "class_index": i + 1,
            "mean": float(label_mean[i]) if not np.isnan(label_mean[i]) else None,
            "std":  float(label_std[i])  if not np.isnan(label_std[i])  else None,
        }
        for i in range(num_labels)
    },
    "per_subject": {
        subj: {
            LABEL_NAMES.get(i+1, f"class_{i+1}"): (
                float(dices[i]) if not np.isnan(dices[i]) else None
            )
            for i in range(num_labels)
        }
        for subj, dices in per_subject_dices.items()
    },
}
with open(VAL_DICE_JSON, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"JSON written to:\n  {VAL_DICE_JSON}")
