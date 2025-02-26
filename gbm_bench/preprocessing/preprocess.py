import os
import argparse
import datetime
from gbm_bench.utils.utils import timed_print
from gbm_bench.preprocessing.dicom_to_nifti import niftiConvert
from gbm_bench.preprocessing.tumor_segmentation import run_brats
from gbm_bench.preprocessing.norm_ss_coregistration import norm_ss_coregister, register_recurrence
from gbm_bench.preprocessing.tissue_segmentation import generate_healthy_brain_mask, run_tissue_seg_registration


def preprocess_dicom(t1: str, t1c: str, t2: str, flair: str, dcm2niix_location: str, pre_treatment: bool, cuda_device: str = "2",
                     perform_nifti_conversion: bool = True, perform_skullstripping: bool = True, perform_tumorseg: bool = True,
                     perform_tissueseg: bool = True) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device


    # Step 1: DICOM to NIfTI conversion
    timed_print("Starting DICOM to NIfTI conversion...")
    dicom_modalities = {
            "t1" : t1,
            "t1c" : t1c,
            "t2" : t2,
            "flair" : flair
            }

    for modality_name, dicom_folder in dicom_modalities.items():
        outdir = os.path.join(os.path.dirname(dicom_folder), "preprocessing")
        nifti_dir = os.path.join(outdir, "nifti_conversion")

        if perform_nifti_conversion:
            niftiConvert(
                    input_dir=dicom_folder,
                    export_dir=nifti_dir,
                    outfile=modality_name,
                    dcm2niix_location=dcm2niix_location
                    )

    timed_print("Finished DICOM to NIfTI conversion.")


    # Step 2: Normalization, co-registration, skull stripping
    timed_print("Starting normalization, co-registration and skull stripping...")
    preprocessed_dir = os.path.join(outdir, "skull_stripped")

    if perform_skullstripping:
        norm_ss_coregister(
                t1=os.path.join(nifti_dir, "t1.nii.gz"),
                t1c=os.path.join(nifti_dir, "t1c.nii.gz"),
                t2=os.path.join(nifti_dir, "t2.nii.gz"),
                flair=os.path.join(nifti_dir, "flair.nii.gz"),
                outdir=preprocessed_dir
                )

    timed_print("Finished normalization, co-registration and skull stripping.")


    # Step 3: Segment tumor
    timed_print("Starting tumor segmentation...")
    tumor_outdir = os.path.join(outdir, "tumor_segmentation")
    tumor_outfile = os.path.join(tumor_outdir, "tumor_seg.nii.gz")
    os.makedirs(tumor_outdir, exist_ok=True)

    if perform_tumorseg:
        run_brats(
                t1=os.path.join(preprocessed_dir, "t1_bet_normalized.nii.gz"),
                t1c=os.path.join(preprocessed_dir, "t1c_bet_normalized.nii.gz"),
                t2=os.path.join(preprocessed_dir, "t2_bet_normalized.nii.gz"),
                flair=os.path.join(preprocessed_dir, "flair_bet_normalized.nii.gz"),
                outfile=tumor_outfile,
                pre_treatment=pre_treatment,
                cuda_device=cuda_device
                )

    timed_print("Finished tumor segmentation.")


    # Step 4: Segment tissue
    timed_print("Starting tissue segmentation...")
    brain_mask_dir = os.path.join(preprocessed_dir, "t1c_bet_mask.nii.gz")
    healthy_mask_dir = os.path.join(tumor_outdir, "healthy_brain_mask.nii.gz")

    generate_healthy_brain_mask(
            brain_mask_file=brain_mask_dir,
            tumor_seg_file=tumor_outfile,
            outdir=healthy_mask_dir
            )

    tissue_seg_dir = os.path.join(outdir, "tissue_segmentation")
    if perform_tissueseg:
        run_tissue_seg_registration(
                t1_file=os.path.join(preprocessed_dir, "t1c_bet_normalized.nii.gz"),
                healthy_mask_dir=healthy_mask_dir,
                brain_mask_dir=brain_mask_dir,
                outdir=tissue_seg_dir,
                refit_brain=False
                )

    timed_print("Finished tissue segmentation.")


def process_longitudinal(preop_exam: str, postop_exam: str) -> None:
    
    t1c_pre_dir = os.path.join(preop_exam, "preprocessing/skull_stripped/t1c_bet_normalized.nii.gz")
    t1c_post_dir = os.path.join(postop_exam, "preprocessing/skull_stripped/t1c_bet_normalized.nii.gz")
    recurrence_seg_dir = os.path.join(postop_exam, "preprocessing/tumor_segmentation/tumor_seg.nii.gz")
    outdir = os.path.join(postop_exam, "preprocessing/longitudinal")

    timed_print("Starting longitudinal registration.")
    register_recurrence(
            t1c_pre_dir=t1c_pre_dir,
            t1c_post_dir=t1c_post_dir,
            recurrence_seg_dir=recurrence_seg_dir,
            outdir=outdir
            )
    timed_print("Finished longitudinal registration.")


if __name__ == "__main__":
    # Example:
    # python gbm_bench/preprocessing/preprocess.py -cuda_device 4
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="4", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    dcm2niix_location = "/home/home/lucas/bin/dcm2niix"

    # Pre-treatment example
    preprocess_dicom(
            t1="test_data/exam1/t1",
            t1c="test_data/exam1/t1c",
            t2="test_data/exam1/t2",
            flair="test_data/exam1/flair",
            dcm2niix_location=dcm2niix_location,
            pre_treatment=True,
            cuda_device=args.cuda_device
            )
    
    # Post-treatment example
    preprocess_dicom(
            t1="test_data/exam3/t1",
            t1c="test_data/exam3/t1c",
            t2="test_data/exam3/t2",
            flair="test_data/exam3/flair",
            dcm2niix_location=dcm2niix_location,
            pre_treatment=False,
            perform_tissueseg=False,
            cuda_device=args.cuda_device
            )

    # Longitudinal example
    process_longitudinal(
            preop_exam="test_data/exam1",
            postop_exam="test_data/exam3"
            )
