import os
import time
import argparse
from pathlib import Path
from loguru import logger
from gbm_bench.preprocessing.dicom_to_nifti import convert_nifti
from gbm_bench.preprocessing.tumor_segmentation import run_brats
from gbm_bench.preprocessing.norm_ss_coregistration import norm_ss_coregister, register_recurrence
from gbm_bench.preprocessing.tissue_segmentation import generate_healthy_brain_mask, run_tissue_seg_registration
from gbm_bench.utils.constants import DCM2NIIX_LOCATION, MODALITY_CONVERTED_SCHEMA, MODALITY_STRIPPED_SHEMA, BRAIN_MASK_SCHEMA, HEALTHY_BRAIN_MASK_SCHEMA, TUMORSEG_SCHEMA


def preprocess_dicom(t1_dir: Path, t1c_dir: Path, t2_dir: Path, flair_dir: Path, pre_treatment: bool, outdir: Path,
                     dcm2niix_location: Path = DCM2NIIX_LOCATION, cuda_device: str = "2", perform_nifti_conversion: bool = True,
                     perform_skullstripping: bool = True, perform_tumorseg: bool = True, perform_tissueseg: bool = True) -> None:
    """
    Performs a multitude of processing steps on raw DICOM data to prepare inputs for tumor growth models. DICOM data is first
    converted to NIfTI, followed by normalization, skull stripping and co-registration on the T1c image. Next, tumor tissue
    is segmented using BRATS algorithms. Finally, tissue segmentation is performed for wm, gm, csf as a multitude of growth
    models require tissue maps as inputs.

    Parameters:
         t1_dir (Path): Path to the directory with the DICOM files for T1.
         t1c_dir (Path): Path to the directory with the DICOM files for T1c.
         t2_dir (Path): Path to the directory with the DICOM files for T2.
         flair_dir (Path): Path to the directory with the DICOM files for Flair.
         pre_treatment (bool): Wether the provided DICOM are preop (True) or postop (False).
             Causes the BRATS segmentation algorithm to choose different models.
         outdir (Path): Base directory for the output. Usually exam directory.
         dcm2niix_location (Path, optional): The location of the dcm2niix executable.
         cuda_device (str): GPU device to use.
         perform_nifti_conversion (bool): Switch for the nifti conversion step.
         perform_skullstripping (bool): Switch for the normalization/skull strip/registration step.
         perform_tumorseg (bool): Switch for the tumor segmentation step.
         perform_tissueseg (bool): Switch for the tissue segmentation step.

    Returns:
        None
    """
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    logger.info("Starting preprocessing.")

    # Step 1: DICOM to NIfTI conversion
    start_time_conversion = time.time()
    logger.info(f"Starting DICOM to NIfTI conversion.")
    dicom_modalities = {
            "t1" : t1_dir,
            "t1c" : t1c_dir,
            "t2" : t2_dir,
            "flair" : flair_dir
            }

    for modality_name, dicom_dir in dicom_modalities.items():
        # Remove suffix since dicom2niix adds .nii.gz automatically
        outfile_tmp = MODALITY_CONVERTED_SCHEMA.format(outdir=outdir, modality=modality_name).with_suffix("").with_suffix("")

        if perform_nifti_conversion:
            convert_nifti(
                    input_dir=dicom_dir,
                    outfile=outfile_tmp,
                    dcm2niix_location=dcm2niix_location
                    )
    time_spent_conversion = time.time() - start_time_conversion
    logger.info(f"Finished conversion step in {time_spent_conversion:.2f} seconds.")

    # Step 2: Normalization, co-registration, skull stripping
    if perform_skullstripping:
        norm_ss_coregister(
                t1_file=MODALITY_CONVERTED_SCHEMA.format(outdir=outdir, modality="t1"),
                t1c_file=MODALITY_CONVERTED_SCHEMA.format(outdir=outdir, modality="t1c"),
                t2_file=MODALITY_CONVERTED_SCHEMA.format(outdir=outdir, modality="t2"),
                flair_file=MODALITY_CONVERTED_SCHEMA.format(outdir=outdir, modality="flair"),
                outdir=outdir
                )

    # Step 3: Segment tumor
    if perform_tumorseg:
        run_brats(
                t1_file=MODALITY_STRIPPED_SHEMA.format(outdir=outdir, modality="t1"),
                t1c_file=MODALITY_STRIPPED_SHEMA.format(outdir=outdir, modality="t1c"),
                t2_file=MODALITY_STRIPPED_SHEMA.format(outdir=outdir, modality="t2"),
                flair_file=MODALITY_STRIPPED_SHEMA.format(outdir=outdir, modality="flair"),
                outdir=outdir,
                pre_treatment=pre_treatment,
                cuda_device=cuda_device
                )

    # Step 4: Segment tissue
    generate_healthy_brain_mask(
            brain_mask_file=BRAIN_MASK_SCHEMA.format(outdir=outdir),
            tumor_seg_file=TUMORSEG_SCHEMA.format(outdir=outdir),
            outfile=HEALTHY_BRAIN_MASK_SCHEMA.format(outdir=outdir)
            )

    if perform_tissueseg:
        run_tissue_seg_registration(
                t1_file = MODALITY_STRIPPED_SHEMA.format(outdir=outdir, modality="t1c"),
                healthy_mask_file=HEALTHY_BRAIN_MASK_SCHEMA.format(outdir=outdir),
                brain_mask_file=BRAIN_MASK_SCHEMA.format(outdir=outdir),
                outdir=outdir,
                refit_brain=False
                )
    time_spent = time.time() - start_time
    logger.info(f"Finished preprocessing in {time_spent:.2f} seconds. Results saved to {outdir}.")


def process_longitudinal(preop_exam: Path, postop_exam: Path, outdir: Path) -> None:
    """
    TODO
    """
    start_time = time.time()
    logger.info(f"Starting longitudinal processing.")

    # Prepare directories
    t1c_pre_file = MODALITY_STRIPPED_SHEMA.format(outdir=preop_exam, modality="t1c")
    t1c_post_file = MODALITY_STRIPPED_SHEMA.format(outdir=postop_exam, modality="t1c")
    recurrence_seg_file = TUMORSEG_SCHEMA.format(outdir=postop_exam)

    register_recurrence(
            t1c_pre_file=t1c_pre_file,
            t1c_post_file=t1c_post_file,
            recurrence_seg_file=recurrence_seg_file,
            outdir=outdir
            )
    time_spent = time.time() - start_time
    logger.info(f"Finished longitudinal preprocessing in {time_spent:.2f} seconds. Results saved to {outdir}.")


if __name__ == "__main__":
    # Example:
    # python gbm_bench/preprocessing/preprocess.py -cuda_device 4
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    outdir_tmp = Path("tmp_preprocessing")

    # Pre-treatment example
    preprocess_dicom(
            t1_dir=Path("test_data/exam1/t1"),
            t1c_dir=Path("test_data/exam1/t1c"),
            t2_dir=Path("test_data/exam1/t2"),
            flair_dir=Path("test_data/exam1/flair"),
            outdir=outdir_tmp / "preop",
            pre_treatment=True,
            cuda_device=args.cuda_device
            )
    
    # Post-treatment example
    preprocess_dicom(
            t1_dir=Path("test_data/exam3/t1"),
            t1c_dir=Path("test_data/exam3/t1c"),
            t2_dir=Path("test_data/exam3/t2"),
            flair_dir=Path("test_data/exam3/flair"),
            outdir=outdir_tmp / "postop",
            pre_treatment=False,
            perform_tissueseg=False,
            cuda_device=args.cuda_device
            )

    # Longitudinal example
    process_longitudinal(
            preop_exam=Path("test_data/exam1"),
            postop_exam=Path("test_data/exam3"),
            outdir=outdir_tmp / "longitudinal"
            )
