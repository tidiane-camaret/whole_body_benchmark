# Training a Patchwork Hierarchical CNN for Medical Image Segmentation

## What is Patchwork?

Patchwork is a TensorFlow framework that trains hierarchical CNNs for medical image segmentation. The key idea: instead of processing a whole image at once (which is memory-prohibitive for 3D volumes), it samples patches at multiple resolution levels and stitches predictions back together.

**Architecture overview:**
```
Full Image
    └─ Level 0 (coarse) → CNN Block 0 → coarse prediction + features
         └─ Level 1 (medium) → CNN Block 1 + Level 0 features → finer prediction
              └─ Level 2 (fine) → CNN Block 2 + Level 1 features → final prediction
```

---

## Installation

```bash
pip install -r requirements.txt
# or
pip install tensorflow==2.12.0 numpy nibabel matplotlib connected-components-3d
pip install -e .
```

---

## Step 1 — Prepare Your Data

Patchwork expects NIFTI files (`.nii` or `.nii.gz`). Organize them as a dict mapping subject IDs to file paths:

```python
import patchwork

# One dict per contrast type (e.g., T1, T2). Here just one contrast.
contrasts = [
    {
        'subj01': '/data/images/subj01_t1.nii.gz',
        'subj02': '/data/images/subj02_t1.nii.gz',
        'subj03': '/data/images/subj03_t1.nii.gz',
    }
]

# One dict per label type. Values are binary masks (0/1).
labels = [
    {
        'subj01': '/data/labels/subj01_mask.nii.gz',
        'subj02': '/data/labels/subj02_mask.nii.gz',
        'subj03': '/data/labels/subj03_mask.nii.gz',
    }
]

subjects = ['subj01', 'subj02', 'subj03']
```

Load the data:

```python
nD = 2  # Use 2 for 2D slices, 3 for volumetric data

tset, lset, rset, subjs = patchwork.improc_utils.load_data_structured(
    contrasts=contrasts,
    labels=labels,
    subjects=subjects,
    nD=nD,
    max_num_data=None,           # Set to N to use only the first N subjects
    threshold=0.5,               # Binarize labels at this threshold
    exclude_incomplete_labels=1, # Skip subjects with missing label files
    align_physical=False,        # Set True to align via the NIFTI affine matrix
)
# tset: list of image arrays, one per subject [H, W, num_contrasts]
# lset: list of label arrays, one per subject [H, W, num_labels]
# rset: list of resolution dicts {'voxsize': ..., 'input_edges': ...}
```

---

## Step 2 — Configure the Patching Strategy

The `patching` dict controls how the hierarchical patch sampling works:

```python
patching = {
    "depth": 3,             # Number of hierarchical levels (2-4)
    "ndim": nD,             # Must match nD above
    "scheme": {
        # Patch size at each level (same for all levels here)
        "patch_size": [32, 32],          # 2D: [H, W]
        # "patch_size": [32, 32, 32],    # 3D: [H, W, D]

        # Output voxel size relative to input (3 = 3x downsampled at coarsest level)
        "destvox_rel": [3, 3],

        # Field of view each patch covers, relative to full image
        "fov_rel": [0.9, 0.9],
    },
    "smoothfac_data": 0,        # Smooth input before patching (0 = no smoothing)
    "smoothfac_label": 0,       # Smooth labels before patching
    "normalize_input": 'mean',  # Normalize patches: 'mean', 'max', None
    "interp_type": "NN",        # Interpolation: "NN" (nearest) or "lin" (linear)
    "scatter_type": "NN",       # How to stitch predictions back together
}
```

**Quick guide for `depth`:**

| depth | Use when |
|-------|----------|
| 2 | Small images, fast experiments |
| 3 | Standard (most use cases) |
| 4 | Large images with fine detail |

---

## Step 3 — Define the Network Architecture

```python
from tensorflow.keras import layers

num_labels = lset[0].shape[nD + 1]   # Last dim of label array = number of classes

network = {
    "num_labels": num_labels,
    "modelname": "models/my_segmentation_model",  # Checkpoint path (auto-saves here)

    # Factory function: called once per hierarchy level to build a U-Net
    "blockCreator": lambda level, outK, input_shape:
        patchwork.customLayers.createUnet_v2(
            depth=5,                          # U-Net encoder/decoder depth
            outK=outK,                        # Output channels (set by PatchWork)
            nD=nD,
            input_shape=input_shape,
            feature_dim=[8, 16, 16, 32, 64]  # Feature channels at each U-Net level
        ),

    # Features passed from coarser -> finer level (enables hierarchical reasoning)
    "intermediate_out": 8,

    # Compute segmentation loss at intermediate levels too (recommended)
    "intermediate_loss": True,

    # Final activation: sigmoid for binary, softmax for multi-class
    "finalBlock": layers.Activation('sigmoid'),
}
```

**Choosing `feature_dim`:**
- Lightweight: `[4, 8, 8, 16, 32]`
- Standard: `[8, 16, 16, 32, 64]`
- High capacity: `[16, 32, 32, 64, 128]` (requires more VRAM)

---

## Step 4 — Build the Model

```python
# Create the hierarchical crop/stitch manager
cgen = patchwork.CropGenerator(**patching)

# Build the full model
themodel = patchwork.PatchWorkModel(cgen, **network)

# Initialize with one example image — this triggers lazy weight creation
example_data = tset[0][0:1, ...]   # Shape: [1, H, W, num_contrasts]
themodel.apply_full(
    example_data,
    resolution=rset[0],
    repetitions=100,
    generate_type='random',
    verbose=True,
)
```

> The `apply_full` call here is not inference — it is needed to trigger TensorFlow's lazy graph building so all weights are created before training starts.

---

## Step 5 — Configure and Run Training

```python
training = {
    "num_patches": 100,       # Patches sampled per image per iteration
    "epochs": 20,             # Gradient steps per patching cycle
    "num_its": 100,           # Total patching cycles (outer loop)
    "reload_after_it": 5,     # Re-sample patches every N iterations
    "samples_per_it": 15,     # Subjects loaded per iteration (memory control)
    "balance": None,          # {"ratio": 0.5, "autoweight": 1} for class imbalance
}

themodel.train(
    tset, lset,
    resolutions=rset,
    **training,
    valid_ids=[2],            # Subject indices to use as validation
    augment={"dphi": 0.1, "flip": [0, 1], "dscale": [0.1, 0.1]},
    batch_size=32,
    verbose=2,
    autosave=True,            # Saves best model to network["modelname"]
    debug=True,
)
```

**What happens inside each iteration:**
1. A subset of subjects is loaded (controlled by `samples_per_it`)
2. `num_patches` patches are sampled at each hierarchy level
3. The U-Nets are trained for `epochs` gradient steps
4. Losses from all hierarchy levels are combined
5. Every `reload_after_it` iterations, patches are re-sampled

---

## Step 6 — Inference on New Images

**From a NumPy array:**
```python
result = themodel.apply_full(
    new_data,                 # [H, W, num_contrasts]
    resolution=rset[0],       # Resolution dict (can reuse from training data)
    num_patches=500,          # More patches = better coverage = slower
    generate_type='random',   # 'random' for most cases, 'grid' for systematic
    verbose=2,
)
# result shape: [H, W, num_labels], values in [0, 1]
binary_mask = result > 0.5
```

**Directly from a NIFTI file:**
```python
themodel.apply_on_nifti(
    'input.nii.gz',
    'output.nii',
    out_typ='mask',           # 'mask' (binary), 'prob' (probabilities)
    repetitions=50,
    num_chunks=10,            # Split into chunks to manage memory
    generate_type='random',
)
```

---

## 3D Training

Switch from 2D to 3D by changing two parameters:

```python
nD = 3   # Was 2

patching["scheme"]["patch_size"] = [32, 32, 32]   # Was [32, 32]
patching["scheme"]["destvox_rel"] = [3, 3, 3]
patching["scheme"]["fov_rel"] = [0.9, 0.9, 0.9]
```

Everything else (model construction, `train()` call, inference) stays the same. 3D requires significantly more memory — reduce `batch_size`, `num_patches`, and `feature_dim` if you hit OOM errors.

---

## Full Runnable Examples

```bash
cd tests/
python example_train2D.py    # 2D segmentation on the bundled test NIFTI files
python example_train3D.py    # 3D version using t1.nii brain scan
```

These scripts use the sample data in `tests/` and are the canonical reference for each configuration option.

---

## Hyperparameter Quick Reference

| Parameter | Conservative | Standard | Aggressive |
|-----------|-------------|----------|------------|
| `depth` | 2 | 3 | 4 |
| `patch_size` (2D) | `[16,16]` | `[32,32]` | `[64,64]` |
| `feature_dim` | `[4,8,8,16]` | `[8,16,16,32,64]` | `[16,32,64,128]` |
| `intermediate_out` | 4 | 8 | 16 |
| `num_patches` | 50 | 100 | 200 |
| `epochs` | 10 | 20 | 50 |
| `num_its` | 50 | 100 | 200 |
| `batch_size` | 16 | 32 | 64 |
