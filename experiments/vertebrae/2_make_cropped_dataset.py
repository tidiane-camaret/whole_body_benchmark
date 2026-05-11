"""Build Dataset002_cropped: 128×128×32-voxel crops centred on TotalSegmentator L1 predictions."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET001 = Path(
    "/nfs/data/nii/data1/Analysis/zanderch___HU_Messung/ANALYSIS_seg/nnunet"
    "/nnUNet_raw/Dataset001_Vertebrae"
)
DATASET002 = Path(
    "/nfs/data/nii/data1/Analysis/zanderch___HU_Messung/ANALYSIS_seg/nnunet"
    "/nnUNet_raw/Dataset002_cropped"
)
CROP_X = 128
CROP_Y = 128
CROP_Z = 32


def _crop_slice(center: int, size: int, dim: int) -> slice:
    half = size // 2
    lo = max(0, min(center - half, dim - size))
    return slice(lo, lo + size)


def crop_case(
    img_path: Path,
    roi_path: Path,
    l1_path: Path,
    out_img_path: Path,
    out_roi_path: Path,
    crop_x: int,
    crop_y: int,
    crop_z: int,
) -> bool:
    img_nib = nib.load(img_path)
    roi_nib = nib.load(roi_path)
    l1_data = nib.load(l1_path).get_fdata()

    coords = np.argwhere(l1_data > 0)
    if len(coords) == 0:
        print(f"  WARNING: empty L1 mask — skipping {img_path.name}")
        return False

    cx, cy, cz = coords.mean(axis=0).astype(int)

    img = img_nib.get_fdata(dtype=np.float32)
    roi = roi_nib.get_fdata()

    sx = _crop_slice(cx, crop_x, img.shape[0])
    sy = _crop_slice(cy, crop_y, img.shape[1])
    sz = _crop_slice(cz, crop_z, img.shape[2])

    img_crop = img[sx, sy, sz]
    roi_crop = roi[sx, sy, sz]

    # Shift the affine origin so voxel [0,0,0] maps to the correct world position.
    # Image and ROI share the same affine, so one update keeps them aligned.
    affine = img_nib.affine.copy()
    affine[:, 3] = affine @ np.array([sx.start, sy.start, sz.start, 1.0])

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_roi_path.parent.mkdir(parents=True, exist_ok=True)

    nib.save(nib.Nifti1Image(img_crop, affine), out_img_path)
    nib.save(nib.Nifti1Image(roi_crop, affine), out_roi_path)
    return True


def main(dataset001: Path, dataset002: Path, crop_x: int, crop_y: int, crop_z: int) -> None:
    imgs_dir      = dataset001 / "imagesTr"
    roi_dir       = dataset001 / "labelsTr"
    vertebrae_dir = dataset001 / "l1_segmentations"

    out_imgs_dir = dataset002 / "imagesTr"
    out_roi_dir  = dataset002 / "labelsTr"

    cases = sorted(
        p.name for p in vertebrae_dir.iterdir()
        if (vertebrae_dir / p.name / "vertebrae_L1.nii.gz").exists()
        and (roi_dir / f"{p.name}.nii.gz").exists()
        and (imgs_dir / f"{p.name}_0000.nii.gz").exists()
    )
    print(f"Found {len(cases)} cases with complete data")

    n_done = 0
    for i, case in enumerate(cases, 1):
        out_img = out_imgs_dir / f"{case}_0000.nii.gz"
        out_roi = out_roi_dir  / f"{case}.nii.gz"

        if out_img.exists() and out_roi.exists():
            print(f"[{i}/{len(cases)}] Skipping (done): {case}")
            n_done += 1
            continue

        print(f"[{i}/{len(cases)}] Cropping: {case}")
        if crop_case(
            imgs_dir / f"{case}_0000.nii.gz",
            roi_dir  / f"{case}.nii.gz",
            vertebrae_dir / case / "vertebrae_L1.nii.gz",
            out_img,
            out_roi,
            crop_x,
            crop_y,
            crop_z,
        ):
            n_done += 1

    dataset002.mkdir(parents=True, exist_ok=True)
    dataset_json = {
        "name": "Dataset002_cropped",
        "description": f"L1-centred {crop_x}×{crop_y}×{crop_z}-voxel crops from Dataset001_Vertebrae",
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "vertebrae_L1": 1},
        "numTraining": n_done,
        "file_ending": ".nii.gz",
    }
    with open(dataset002 / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDone. {n_done}/{len(cases)} cases written to {dataset002}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset001", type=Path, default=DATASET001)
    parser.add_argument("--dataset002", type=Path, default=DATASET002)
    parser.add_argument("--crop_x", type=int, default=CROP_X)
    parser.add_argument("--crop_y", type=int, default=CROP_Y)
    parser.add_argument("--crop_z", type=int, default=CROP_Z)
    args = parser.parse_args()
    main(args.dataset001, args.dataset002, args.crop_x, args.crop_y, args.crop_z)
