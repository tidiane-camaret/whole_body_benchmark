"""Segment L1 vertebra for all CT NIfTI files using TotalSegmentator."""

import argparse
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator

INPUT_DIR = Path(
    "/nfs/data/nii/data1/Analysis/zanderch___HU_Messung/ANALYSIS_seg/nnunet"
    "/nnUNet_raw/Dataset001_Vertebrae/imagesTr"
)
DEFAULT_OUTPUT_DIR = Path(
    "/nfs/data/nii/data1/Analysis/zanderch___HU_Messung/ANALYSIS_seg/nnunet"
    "/nnUNet_raw/Dataset001_Vertebrae/l1_segmentations"
)


def segment_l1(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.nii.gz"))
    print(f"Found {len(files)} files in {input_dir}")

    for i, nii_path in enumerate(files, 1):
        # Output: one folder per case, TotalSegmentator writes vertebrae_L1.nii.gz inside
        case_out = output_dir / nii_path.name.replace("_0000.nii.gz", "")
        done_marker = case_out / "vertebrae_L1.nii.gz"

        if done_marker.exists():
            print(f"[{i}/{len(files)}] Skipping (already done): {nii_path.name}")
            continue

        print(f"[{i}/{len(files)}] Segmenting: {nii_path.name}")
        try:
            totalsegmentator(
                input=nii_path,
                output=case_out,
                task="total",
                roi_subset=["vertebrae_L1"],
                device="gpu",
                fast=False,
            )
            print(f"  -> Saved to {case_out}")
        except Exception as e:
            print(f"  ERROR on {nii_path.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment L1 vertebra from CT NIfTI files.")
    parser.add_argument("--input_dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    segment_l1(args.input_dir, args.output_dir)
