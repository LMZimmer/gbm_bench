import os
import argparse
import numpy as np
import nibabel as nib
from brats import AdultGliomaPreTreatmentSegmenter
from brats import AdultGliomaPostTreatmentSegmenter
from brats.constants import AdultGliomaPreTreatmentAlgorithms
from brats.constants import AdultGliomaPostTreatmentAlgorithms


def split_segmentation(tumor_seg_file: str) -> None:
    # 1: non_enhancing, 2: edema, 3: enhancing
    outdir = os.path.dirname(tumor_seg_file)
    tumor_seg = nib.load(tumor_seg_file)

    enhancing_non_enhancing = nib.Nifti1Image(
            ((tumor_seg.get_fdata() == 1) | (tumor_seg.get_fdata() == 3)).astype(np.int32),
            header=tumor_seg.header,
            affine=tumor_seg.affine)
    
    edema = nib.Nifti1Image(
            (tumor_seg.get_fdata() == 2).astype(np.int32),
            header=tumor_seg.header,
            affine=tumor_seg.affine)

    nib.save(enhancing_non_enhancing, os.path.join(outdir, "enhancing_non_enhancing_tumor.nii.gz"))
    nib.save(edema, os.path.join(outdir, "peritumoral_edema.nii.gz"))


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
