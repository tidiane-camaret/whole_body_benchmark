import sys
import numpy as np
import nibabel as nib
import edt
from scipy.ndimage import binary_erosion

def compute_surface(mask):
    """Surface voxels = mask minus its erosion."""
    eroded = binary_erosion(mask, border_value=1)
    return mask & ~eroded

def dice(gt_k, pred_k):
    intersection = np.sum(gt_k & pred_k)
    total = np.sum(gt_k) + np.sum(pred_k)
    if total == 0:
        return float("nan")
    return 2 * intersection / total

def nsd(gt_k, pred_k, voxel_spacing, tolerance_mm=2.0):
    """Normalized Surface Dice at given tolerance (mm)."""
    surf_gt   = compute_surface(gt_k)
    surf_pred = compute_surface(pred_k)
    n_surf = surf_gt.sum() + surf_pred.sum()
    if n_surf == 0:
        return float("nan")

    # distance from every voxel to the surface of the other structure
    dist_from_gt   = edt.edt(~surf_gt,   anisotropy=voxel_spacing)
    dist_from_pred = edt.edt(~surf_pred, anisotropy=voxel_spacing)

    within = (surf_pred & (dist_from_gt   <= tolerance_mm)).sum() \
           + (surf_gt   & (dist_from_pred <= tolerance_mm)).sum()
    return within / n_surf


def main(gt_path, pred_path, tolerance_mm=2.0):
    gt_nii   = nib.load(gt_path)
    pred_nii = nib.load(pred_path)

    gt   = np.asarray(gt_nii.dataobj, dtype=np.int32)
    pred = np.asarray(pred_nii.dataobj, dtype=np.int32)

    print(f"GT shape:   {gt.shape}")
    print(f"Pred shape: {pred.shape}")

    # label map is channel 0 for patchwork 'atls' output
    if pred.ndim == 4:
        print("Using pred[..., 0] as label map")
        pred = pred[..., 0]

    voxel_spacing = gt_nii.header.get_zooms()[:3]
    print(f"Voxel spacing (mm): {voxel_spacing}")
    print(f"NSD tolerance: {tolerance_mm} mm\n")

    labels = sorted(set(np.unique(gt).tolist() + np.unique(pred).tolist()))
    labels = [l for l in labels if l != 0]

    print(f"{'Label':>6}  {'Dice':>8}  {'NSD':>8}")
    print("-" * 28)

    dice_vals, nsd_vals = [], []
    for k in labels:
        gt_k   = gt   == k
        pred_k = pred == k
        d = dice(gt_k, pred_k)
        n = nsd(gt_k, pred_k, voxel_spacing, tolerance_mm)
        dice_vals.append(d)
        nsd_vals.append(n)
        d_str = f"{d:.4f}" if not np.isnan(d) else "   nan"
        n_str = f"{n:.4f}" if not np.isnan(n) else "   nan"
        print(f"{k:>6}  {d_str:>8}  {n_str:>8}")

    valid_dice = [v for v in dice_vals if not np.isnan(v)]
    valid_nsd  = [v for v in nsd_vals  if not np.isnan(v)]
    print("-" * 28)
    print(f"{'mean':>6}  {np.mean(valid_dice):>8.4f}  {np.mean(valid_nsd):>8.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compute_metrics.py <gt.nii.gz> <pred.nii.gz> [tolerance_mm]")
        sys.exit(1)
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    main(sys.argv[1], sys.argv[2], tol)
