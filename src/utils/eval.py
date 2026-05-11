import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label as nd_label
import sys
sys.path.insert(0, "/home/dpxuser/dev/whole_body_benchmark")
from src.compute_metrics import dice


def eval_case(img_path, pred_path, gt_path, label=1):
    img_nii  = nib.load(img_path)
    pred_nii = nib.load(pred_path)
    gt_nii   = nib.load(gt_path)

    img  = np.asarray(img_nii.dataobj,  dtype=np.float32)
    pred = np.asarray(pred_nii.dataobj, dtype=np.int32)
    gt   = np.asarray(gt_nii.dataobj,   dtype=np.int32)

    spacing = np.array(gt_nii.header.get_zooms()[:3])

    pred_mask = pred == label
    gt_mask   = gt   == label

    # Dice
    dice_score = dice(gt_mask, pred_mask)

    # Dice on largest connected component of pred
    labeled, n = nd_label(pred_mask)
    if n > 0:
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        largest = labeled == (np.argmax(sizes) + 1)
        dice_lcc = dice(gt_mask, largest)
    else:
        dice_lcc = float("nan")

    # Center of mass distance in mm
    def center_of_mass(mask):
        coords = np.array(np.where(mask), dtype=float)
        return coords.mean(axis=1) if coords.size > 0 else None

    gt_center   = center_of_mass(gt_mask)
    pred_center = center_of_mass(pred_mask)

    if gt_center is not None and pred_center is not None:
        center_dist_mm = np.linalg.norm((gt_center - pred_center) * spacing)
    else:
        center_dist_mm = float("nan")

    print(f"Dice:              {dice_score:.4f}")
    print(f"Dice (largest CC): {dice_lcc:.4f}")
    print(f"Center distance:   {center_dist_mm:.2f} mm")

    # Slices centered on GT centroid
    if gt_center is not None:
        cx, cy, cz = (int(round(c)) for c in gt_center)
    else:
        cx, cy, cz = (s // 2 for s in img.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slices = [
        ("Axial (z)",    img[:, :, cz],  gt_mask[:, :, cz],  pred_mask[:, :, cz]),
        ("Coronal (y)",  img[:, cy, :],  gt_mask[:, cy, :],  pred_mask[:, cy, :]),
        ("Sagittal (x)", img[cx, :, :],  gt_mask[cx, :, :],  pred_mask[cx, :, :]),
    ]
    for ax, (title, img_sl, gt_sl, pred_sl) in zip(axes, slices):
        vmin, vmax = np.percentile(img_sl, [1, 99])
        ax.imshow(img_sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        for sl, color in [(gt_sl, [0, 1, 0, 0.4]), (pred_sl, [1, 0, 0, 0.4])]:
            overlay = np.zeros((*sl.shape, 4))
            overlay[sl] = color
            ax.imshow(overlay.transpose(1, 0, 2), origin="lower")
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle(f"GT=green  Pred=red  |  Dice={dice_score:.3f}  LCC={dice_lcc:.3f}  Δcenter={center_dist_mm:.1f}mm")
    plt.tight_layout()
    plt.show()

    return {"dice": dice_score, "dice_lcc": dice_lcc, "center_dist_mm": center_dist_mm}
