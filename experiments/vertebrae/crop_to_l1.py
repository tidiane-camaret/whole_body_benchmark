"""Crop an image to a fixed window centred on an L1 mask and save it."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

CROP_X, CROP_Y, CROP_Z = 128, 128, 32


def _crop_slice(center: int, size: int, dim: int) -> slice:
    half = size // 2
    lo = max(0, min(center - half, dim - size))
    return slice(lo, lo + size)


def main(img_path: Path, l1_path: Path, out_path: Path,
         crop_x: int, crop_y: int, crop_z: int) -> None:
    img_nib = nib.load(img_path)
    l1_data = nib.load(l1_path).get_fdata()

    coords = np.argwhere(l1_data > 0)
    if len(coords) == 0:
        raise ValueError(f"Empty L1 mask: {l1_path}")

    cx, cy, cz = coords.mean(axis=0).astype(int)
    img = img_nib.get_fdata(dtype=np.float32)

    sx = _crop_slice(cx, crop_x, img.shape[0])
    sy = _crop_slice(cy, crop_y, img.shape[1])
    sz = _crop_slice(cz, crop_z, img.shape[2])

    affine = img_nib.affine.copy()
    affine[:, 3] = affine @ np.array([sx.start, sy.start, sz.start, 1.0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(img[sx, sy, sz], affine), out_path)
    print(f"Saved {out_path}  shape={img[sx, sy, sz].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("img",    type=Path, help="Input image (.nii.gz)")
    parser.add_argument("l1",     type=Path, help="L1 mask (.nii.gz)")
    parser.add_argument("output", type=Path, help="Output cropped image (.nii.gz)")
    parser.add_argument("--crop_x", type=int, default=CROP_X)
    parser.add_argument("--crop_y", type=int, default=CROP_Y)
    parser.add_argument("--crop_z", type=int, default=CROP_Z)
    args = parser.parse_args()
    main(args.img, args.l1, args.output, args.crop_x, args.crop_y, args.crop_z)
