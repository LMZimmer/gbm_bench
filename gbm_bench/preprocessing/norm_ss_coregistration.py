import os
import ants
import shutil
import argparse
from typing import List, Tuple
from brainles_preprocessing.normalization import Normalizer
from brainles_preprocessing.preprocessor import Preprocessor
from brainles_preprocessing.modality import Modality, CenterModality
from brainles_preprocessing.normalization.percentile_normalizer import PercentileNormalizer


def initialize_center_modality(modality_path: str, modality_name: str, normalizer: Normalizer, outdir: str) -> CenterModality:
    
    bet_out = os.path.join(outdir, "_".join((modality_name, "bet_normalized.nii.gz")))
    bet_mask_out = os.path.join(outdir, "_".join((modality_name, "bet_mask.nii.gz")))

    center = CenterModality(
            modality_name=modality_name,
            input_path=modality_path,
            normalizer=normalizer,
            normalized_bet_output_path=bet_out,
            bet_mask_output_path=bet_mask_out,
            )
    
    return center


def initialize_moving_modalities(modality_paths: List[str], modality_names: List[str], normalizer: Normalizer, outdir: str) -> Modality:
    
    moving_modalities = []
    for mod_path, mod_name in zip(modality_paths, modality_names):
        
        bet_norm_out = os.path.join(outdir, "_".join((mod_name, "bet_normalized.nii.gz")))
        
        m = Modality(
                input_path=mod_path,
                modality_name=mod_name,
                normalizer=normalizer,
                normalized_bet_output_path=bet_norm_out,
                )

        moving_modalities.append(m)
    return moving_modalities


def norm_ss_coregister(t1: str, t1c: str, t2: str, flair: str, outdir: str) -> None:

    percentile_normalizer = PercentileNormalizer(
            lower_percentile=0.1,
            upper_percentile=99,
            lower_limit=0,
            upper_limit=1,
            )

    center = initialize_center_modality(
            modality_path=t1c,
            modality_name="t1c",
            normalizer=percentile_normalizer,
            outdir=outdir
            )
    moving = initialize_moving_modalities(
            modality_paths=[t1, t2, flair],
            modality_names=["t1", "t2", "flair"],
            normalizer=percentile_normalizer,
            outdir=outdir
            )

    preprocessor = Preprocessor(
            center_modality=center,
            moving_modalities=moving,
            )

    #preprocessor.run(save_dir_atlas_registration=output_path / "atlas_registration")
    preprocessor.run()


def register_recurrence(t1c_pre_dir: str, t1c_post_dir: str, recurrence_seg_dir: str, outdir: str) -> None:
    t1c_pre = ants.image_read(t1c_pre_dir)
    t1c_post = ants.image_read(t1c_post_dir)

    reg = ants.registration(
            fixed=t1c_pre,
            moving=t1c_post,
            type_of_transform="SyN",
            reg_iterations=(50, 20),
            shrink_factors=(2, 1),
            smoothing_sigmas=(1, 0)
            )

    os.makedirs(outdir, exist_ok=True)
    shutil.copyfile(src=reg["fwdtransforms"][0], dst=os.path.join(outdir, "longitudinal_trafo.mat"))
    ants.image_write(reg["warpedmovout"], os.path.join(outdir, "t1c_warped_longitudinal.nii.gz"))

    recurrence_seg = ants.image_read(recurrence_seg_dir)
    recurrence_warped = ants.apply_transforms(
            fixed=t1c_pre,
            moving=recurrence_seg,
            transformlist=reg["fwdtransforms"],
            interpolator='nearestNeighbor'
            )
    ants.image_write(recurrence_warped, os.path.join(outdir, "recurrence_preop.nii.gz"))


if __name__ == "__main__":
    # Example:
    # python gbm_bench/preprocessing/norm_ss_coregistration.py -cuda_device 4
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="4", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    t1_nifti = "test_data/exam1/preprocessing/nifti_conversion/t1.nii.gz"
    t1c_nifti = "test_data/exam1/preprocessing/nifti_conversion/t1c.nii.gz"
    t2_nifti = "test_data/exam1/preprocessing/nifti_conversion/t2.nii.gz"
    flair = "test_data/exam1/preprocessing/nifti_conversion/flair.nii.gz"

    t1c_preop_dir = "test_data/exam1/preprocessing/skull_stripped/t1c_bet_normalized.nii.gz"
    t1c_postop_dir = "test_data/exam3/preprocessing/skull_stripped/t1c_bet_normalized.nii.gz"
    recurrence_seg_dir = "test_data/exam3/preprocessing/tumor_segmentation/tumor_seg.nii.gz"

    norm_ss_coregister(
            t1=t1_nifti,
            t1c=t1c_nifti,
            t2=t2_nifti,
            flair=flair,
            outdir="./tmp_test_ss"
            )

    register_recurrence(
            t1c_pre_dir=t1c_preop_dir,
            t1c_post_dir=t1c_postop_dir,
            recurrence_seg_dir=recurrence_seg_dir,
            outdir="test_data/exam3/preprocessing/longitudinal"
            )
