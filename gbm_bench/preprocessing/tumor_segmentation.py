import os
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from loguru import logger
from brats import AdultGliomaPreTreatmentSegmenter
from brats import AdultGliomaPostTreatmentSegmenter
from brats.constants import AdultGliomaPreTreatmentAlgorithms
from brats.constants import AdultGliomaPostTreatmentAlgorithms


def split_segmentation(tumor_seg_file: Path, necrotic_label: int = 1, edema_label: int = 2, enhancing_label: int = 3) -> None:
    """
    Split a composite tumor segmentation into separate binary segmentation files
    for enhancing/non-enhancing tumor and peritumoral edema.

    Parameters:
        tumor_seg_file (Path): Path to the input tumor segmentation NIfTI file.
        necrotic_label (int): Label for necrotic / non-enhancing tissue in the segmentation.
        edema_label (int): Label for edema in the segmentation.
        enhancing_label (int): Label for enhancing tumor in the segmentation.

    Returns:
        None
    """
    outdir = tumor_seg_file.parent
    tumor_seg = nib.load(str(tumor_seg_file))
    seg_data = tumor_seg.get_fdata()

    # Create a binary mask for non-enhancing and enhancing tumor (labels 1 and 3).
    enhancing_non_enhancing = nib.Nifti1Image(
        ((seg_data == necrotic_label) | (seg_data == enhancing_label)).astype(np.int32),
        header=tumor_seg.header,
        affine=tumor_seg.affine
    )

    # Create a binary mask for edema (label 2).
    edema = nib.Nifti1Image(
        (seg_data == edema_label).astype(np.int32),
        header=tumor_seg.header,
        affine=tumor_seg.affine
    )

    nib.save(enhancing_non_enhancing, str(outdir / "enhancing_non_enhancing_tumor.nii.gz"))
    nib.save(edema, str(outdir / "peritumoral_edema.nii.gz"))


def run_brats(t1: str, t1c: str, t2: str, flair: str, outfile: str, pre_treatment: bool = True, cuda_device: str = "4") -> None:

    if pre_treatment:
        segmenter = AdultGliomaPreTreatmentSegmenter(
                algorithm=AdultGliomaPreTreatmentAlgorithms.BraTS23_1,
                cuda_devices=cuda_device
                )
    else:
        segmenter = AdultGliomaPostTreatmentSegmenter(
                algorithm=AdultGliomaPostTreatmentAlgorithms.BraTS24_1,
                cuda_devices=cuda_device
                )

    segmenter.infer_single(
            t1n=t1,
            t1c=t1c,
            t2w=t2,
            t2f=flair,
            output_file=outfile)

    split_segmentation(outfile)


if __name__ == "__main__":
    # Example
    # python gbm_bench/preprocessing/tumor_segmentation.py -cuda_device 4
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="4", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    t1 = "test_data/exam1/preprocessing/skull_stripped/t1_bet_normalized.nii.gz"
    t1c = "test_data/exam1/preprocessing/skull_stripped/t1c_bet_normalized.nii.gz"
    t2 = "test_data/exam1/preprocessing/skull_stripped/t2_bet_normalized.nii.gz"
    flair = "test_data/exam1/preprocessing/skull_stripped/flair_bet_normalized.nii.gz"
    outfiles = "tmp_test_tumorseg/tumor_seg.nii.gz"

    run_brats(
            t1=t1,
            t1c=t1c,
            t2=t2,
            flair=flair,
            outfile=outfile,
            cuda_device=args.cuda_device
            )
