import argparse
import os
from brainles_preprocessing.modality import Modality, CenterModality
from brainles_preprocessing.normalization.percentile_normalizer import (
    PercentileNormalizer, 
)
from brainles_preprocessing.preprocessor import Preprocessor


def initialize_center_modality(modality_path, modality_name, normalizer, outdir):
    
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


def initialize_moving_modalities(modality_paths, modality_names, normalizer, outdir):
    
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


def run_preprocessing(t1, t1c, t2, flair, outdir):

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


if __name__ == "__main__":
    # Example:
    # python gbm_bench/preprocessing/norm_ss_coregistration.py -t1 test_data/exam1/preprocessing/nifti_conversion/t1.nii.gz -t1c test_data/exam1/preprocessing/nifti_conversion/t1c.nii.gz -t2 test_data/exam1/preprocessing/nifti_conversion/t2.nii.gz -flair test_data/exam1/preprocessing/nifti_conversion/flair.nii.gz -outdir tmp_ss_test/
    parser = argparse.ArgumentParser()
    parser.add_argument("-t1", type=str, help="Path to T1 nifti.")
    parser.add_argument("-t1c", type=str, help="Path to T1 contrast nifti.")
    parser.add_argument("-t2", type=str, help="Path to T2 nifti.")
    parser.add_argument("-flair", type=str, help="Path to flair nifti.")
    parser.add_argument("-outdir", type=str, help="Output directory.")
    parser.add_argument("-gpu_device", type=str, default="1", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    
    run_preprocessing(
            t1=args.t1,
            t1c=args.t1c,
            t2=args.t2,
            flair=args.flair,
            outdir=args.outdir
            )
