#!/usr/bin/env python3
"""
Standalone reproduction of the "oppscreen" patchwork 3D training job.
Subjects and paths are resolved from the project Hydra config + splits file.
"""


import sys
import os
import json
import math
import gc
from pathlib import Path
from glob import glob

#sys.path.insert(0, "/software")
sys.path.insert(0, "/nfs/norasys/notebooks/camaret/repos/patchwork")

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# ─── CONFIG & SPLITS ───────────────────────────────────────────────────────────

config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs")
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="config")
print(OmegaConf.to_yaml(cfg))

img_base  = Path(cfg.paths.nako_dir) / "links"
mask_base = Path(cfg.paths.nako_dir) / "data"
img_glob  = "30/3D_GRE_TRA_4/3D_GRE_TRA_W_COMPOSE*_s*.nii"
mask_rel  = "30/opportunistic-screening/seg.nii.gz"

splits_path = Path(cfg.paths.data_dir) / "splits_966.json"
with open(splits_path) as f:
    splits = json.load(f)
subjects_train = splits["train"]
subjects_test  = splits["test"]

MODEL_DIR = str(Path(cfg.paths.results_dir) / "patchwork" / "subsetFW")

# ─── RESOLVE FILE PATHS ────────────────────────────────────────────────────────

def resolve_subject(sid, split_label):
    img_matches = sorted(glob(str(img_base / sid / img_glob)))
    mask_path   = mask_base / sid / mask_rel
    if not img_matches:
        print(f"  [SKIP {split_label}] {sid}: no image found for {img_glob}")
        return None, None
    if not mask_path.exists():
        print(f"  [SKIP {split_label}] {sid}: mask not found at {mask_path}")
        return None, None
    return str(img_matches[0]), str(mask_path)   # "take first match"

INPUT_FILES, TARGET_FILES, SUBJECT_IDS, VALIDATION_IDS = [], [], [], []

for sid in subjects_train:
    img, mask = resolve_subject(sid, "train")
    if img is not None:
        INPUT_FILES.append(img)
        TARGET_FILES.append(mask)
        SUBJECT_IDS.append(sid)

for sid in subjects_test:
    img, mask = resolve_subject(sid, "test")
    if img is not None:
        INPUT_FILES.append(img)
        TARGET_FILES.append(mask)
        SUBJECT_IDS.append(sid)
        VALIDATION_IDS.append(sid)

print(f"Training subjects : {len(subjects_train)} requested, "
      f"{len(SUBJECT_IDS) - len(VALIDATION_IDS)} resolved")
print(f"Validation subjects: {len(subjects_test)} requested, "
      f"{len(VALIDATION_IDS)} resolved")

# ─── PATCHWORK LIBRARY PATH ────────────────────────────────────────────────────
# Adjust if patchwork2 is not already on sys.path
# sys.path.insert(0, "/software/patchwork_master")

# ───────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import numpy as np  # noqa: F401 (used by patchwork star-imports)
import tensorflow as tf
from tensorflow.keras import layers

import patchwork2
from patchwork2 import improc_utils
from patchwork2.customLayers import *
from patchwork2.improc_utils import *
import patchwork2.model as patchwork

import multiprocessing
try:
    multiprocessing.set_start_method('forkserver')
except RuntimeError:
    pass

# ─── GPU SETUP ─────────────────────────────────────────────────────────────────

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"GPU available: {gpus[0]}")
else:
    print("No GPU found, running on CPU")

# ─── PARAMETERS (from job JSON) ────────────────────────────────────────────────

nD = 3

patching = {
    "depth": 4,
    "scheme": {
        "patch_size":   [32, 32, 32],
        "destvox_mm":   [2.0, 2.0, 2.0],
        "destvox_rel":  None,
        "fov_mm":       [300.0, 200.0, 500.0],
        "fov_rel":      None,
    },
    "smoothfac_data":   0,
    "smoothfac_label":  0,
    "interp_type":      "NN",
    "scatter_type":     "NN",
    "normalize_input":  "patch_m0s1",   # "unit patch" in GUI
    "categorial_label": list(range(1, 26)),  # 25-class segmentation
    "categorical":      True,
    "ndim":             nD,
    "system":           "world",
}

network = {
    "blockCreator": lambda level, outK, input_shape: createUnet_v2(
        depth=5, outK=outK, nD=nD, input_shape=input_shape,
        feature_dim=[32, 32, 64, 64, 128]
    ),
    "intermediate_out":  8,
    "intermediate_loss": True,
    "num_labels":        len(patching["categorial_label"]),  # 25
    "finalBlock":        layers.Activation("softmax"),
}

loading = {
    "nD":                       nD,
    "crop_fdim":                [2, 3],   # keep only channels 2 and 3 of input
    "crop_fdim_labels":         None,
    "crop_only_nonzero":        False,
    "threshold":                None,     # overridden by integer_labels
    "integer_labels":           True,     # required for categorical loss
    "exclude_incomplete_labels": 1,       # "skip examples" in GUI
    "annotations_selector":     None,
}

training = {
    "num_patches":    32,
    "num_its":        10000,              # in PK (thousands-of-patches) units
    "epochs":         4,
    "samples_per_it": 200,
    "reload_after_it": 100,
    "maxpatch_per_it": 3200,
    "augment":   {"dphi": 0.3, "flip": [0, 0, 0], "dscale": [0.3, 0.3, 0.3], "vscale": 0.5},
    "balance":   {"ratio": 0.8},
    "fit_type":  "custom",
    "parallel":  "thread",
}

loss = tf.losses.sparse_categorical_crossentropy  # "categorical crossentropy" in GUI
VALIDATION_TAG = "mjvalset"

# ─── BUILD SUBJECT DICTS ───────────────────────────────────────────────────────

assert len(INPUT_FILES) == len(TARGET_FILES) == len(SUBJECT_IDS), \
    "INPUT_FILES, TARGET_FILES and SUBJECT_IDS must have the same length"

# patchwork expects contrasts/labels as list-of-dicts: [{subj_id: filepath, ...}]
key = lambda sid: "subj_" + sid

contrasts = [{key(sid): fp for sid, fp in zip(SUBJECT_IDS, INPUT_FILES)}]
labels    = [{key(sid): fp for sid, fp in zip(SUBJECT_IDS, TARGET_FILES)}]
all_subjects = [key(sid) for sid in SUBJECT_IDS]
num_subjects = len(all_subjects)

def get_data(n=None):
    tset, lset, rset, subjs = load_data_structured(
        contrasts=contrasts,
        labels=labels,
        subjects=all_subjects,
        max_num_data=n,
        **loading,
    )
    if len(tset) == 0:
        raise RuntimeError("No valid data found!")
    return tset, lset, rset, subjs

# ─── MODEL INIT ────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
modelfi   = os.path.join(MODEL_DIR, "model_patchwork")

print("\n>>> Loading one example for network initialisation")
with tf.device("/cpu:0"):
    tset, lset, rset, _ = get_data(1)

network["input_fdim"] = tset[0].shape[-1]
print(f"input_fdim: {network['input_fdim']}")

if os.path.exists(modelfi + ".json"):
    print("\n>>> Model exists — resuming")
    themodel = patchwork.PatchWorkModel.load(modelfi, immediate_init=True, notmpfile=True)
else:
    print("\n>>> Creating new model")
    cgen     = patchwork.CropGenerator(**patching)
    themodel = patchwork.PatchWorkModel(cgen, modelname=modelfi, **network)
    themodel.apply_full(
        tset[0], resolution=rset[0],
        repetitions=1, generate_type="random",
        scale_to_original=False, sampling_factor=0.2,
        verbose=True, init=True,
    )

themodel.summary()

# ─── CONVERT num_its FROM PK → ACTUAL ITERATIONS ──────────────────────────────

reload_after_it  = training.pop("reload_after_it")
samples_per_it   = training.pop("samples_per_it")
maxpatch_per_it  = training.pop("maxpatch_per_it")

planned_patches  = training["num_its"] * 1000             # 10_000_000

subset_size = maxpatch_per_it // training["num_patches"]   # 3200 // 32 = 100
if subset_size >= num_subjects:
    subset_size = None
    num_train_subjects = num_subjects
else:
    num_train_subjects = subset_size

patches_per_iter = training["num_patches"] * num_train_subjects * training["epochs"]
training["num_its"] = round(planned_patches / patches_per_iter)
print(f"Planned patches: {planned_patches:,}  →  {training['num_its']} iterations per reload block")

outer_num_its = math.ceil(training["num_its"] / reload_after_it * num_subjects / samples_per_it)
training["num_its"] = reload_after_it
print(f"Outer reload iterations: {outer_num_its}")

if themodel.train_cycle is None:
    themodel.train_cycle = 0
themodel.train_cycle += 1

# ─── VALIDATION HELPER ────────────────────────────────────────────────────────

def get_valid_ids(subjs):
    return [i for i, s in enumerate(subjs) if s.replace("subj_", "") in VALIDATION_IDS]

# ─── TRAINING LOOP ─────────────────────────────────────────────────────────────

def on_save(numit):
    if numit <= 1:
        saved = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR)]
        print(f"Saved files: {saved}")

print("\n\n>>> Starting training")

initial_patches = themodel.myhist.getNumberOfSeenPatches()

for i in range(outer_num_its):
    print(f"\n=== Reload block {i+1}/{outer_num_its}: loading {samples_per_it} subjects ===")

    with tf.device("/cpu:0"):
        tset, lset, rset, subjs = get_data(samples_per_it)

    valid_ids = get_valid_ids(subjs)
    print(f"  Validation subjects in this batch: {len(valid_ids)}")

    if len(tset) == 0:
        continue

    themodel.train(
        tset, lset,
        resolutions=rset,
        loss=loss,
        valid_ids=valid_ids,
        subset_size=subset_size,
        lazyTrain=None,
        unlabeled_ids=[],
        inc_train_cycle=False,
        callback=on_save,
        verbose=2,
        **training,
    )

    with tf.device("/cpu:0"):
        del tset, lset
        gc.collect()

    if themodel.myhist.getNumberOfSeenPatches() - initial_patches > planned_patches:
        print("Patch budget reached, stopping.")
        break

print("\n>>> Training complete")
print(f"Model saved to: {MODEL_DIR}")
