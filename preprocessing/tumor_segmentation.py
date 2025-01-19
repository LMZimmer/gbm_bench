import argparse
import os
from brats import AdultGliomaSegmenter
from brats.utils.constants import AdultGliomaAlgorithms


def run_brats(t1, t1c, t2, flair, outfile):
    segmenter = AdultGliomaSegmenter(
            algorithm=AdultGliomaAlgorithms.BraTS23_1,
            cuda_devices=CUDA_DEVICES
            )

    segmenter.infer_single(
            t1n=t1,
            t1c=t1c,
            t2=t2,
            t2f=flair,
            output_file=outfile)


if __name__ == "__main__":
    # python 3_tumor_segmentation.py -t1 /home/home/lucas/scripts/test/stripped/t1_bet_normalized.nii.gz -t1c /home/home/lucas/scripts/test/stripped/t1c_bet_normalized.nii.gz -t2 /home/home/lucas/scripts/test/stripped/t2_bet_normalized.nii.gz -flair /home/home/lucas/scripts/test/stripped/flair_bet_normalized.nii.gz -outfile /home/home/lucas/scripts/test/stripped/segmentations/brats_segmentation.nii.gz
    parser = argparse.ArgumentParser()
    parser.add_argument("-t1", type=str, help="Path to T1 nifti.")
    parser.add_argument("-t1c", type=str, help="Path to T1 nifti.")
    parser.add_argument("-t2", type=str, help="Path to T1 nifti.")
    parser.add_argument("-flair", type=str, help="Path to T1 nifti.")
    parser.add_argument("-outfile", type=str, help="Desired file path for output segmentation.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"


    run_brats(t1, t1c, t2, flair, outfile)
