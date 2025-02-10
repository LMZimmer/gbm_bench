import argparse
import os
import numpy as np
import nibabel as nib
from brats import AdultGliomaPreTreatmentSegmenter
from brats.constants import AdultGliomaPreTreatmentAlgorithms
from brats import AdultGliomaPostTreatmentSegmenter
from brats.constants import AdultGliomaPostTreatmentAlgorithms


def split_segmentation(tumor_seg_file):
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


def run_brats(t1, t1c, t2, flair, outfile, pre_treatment=True, cuda_device="2"):

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
    # python tumor_segmentation.py -t1 /home/home/lucas/scripts/test/stripped/t1_bet_normalized.nii.gz -t1c /home/home/lucas/scripts/test/stripped/t1c_bet_normalized.nii.gz -t2 /home/home/lucas/scripts/test/stripped/t2_bet_normalized.nii.gz -flair /home/home/lucas/scripts/test/stripped/flair_bet_normalized.nii.gz -outfile /home/home/lucas/scripts/test/stripped/segmentations/brats_segmentation.nii.gz
    parser = argparse.ArgumentParser()
    parser.add_argument("-t1", type=str, help="Path to T1 nifti.")
    parser.add_argument("-t1c", type=str, help="Path to T1 nifti.")
    parser.add_argument("-t2", type=str, help="Path to T1 nifti.")
    parser.add_argument("-flair", type=str, help="Path to T1 nifti.")
    parser.add_argument("-outfile", type=str, help="Desired file path for output segmentation.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"


    run_brats(t1, t1c, t2, flair, outfile, cuda_device="2")
