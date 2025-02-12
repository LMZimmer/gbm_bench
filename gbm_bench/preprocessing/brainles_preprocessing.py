import argparse
import os
from pathlib import Path
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
            #lower_percentile=0.001,
            #upper_percentile=100,
            #lower_limit=0,
            #upper_limit=1,
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
    # python 2_brainles_preprocess.py -t1 /home/home/lucas/scripts/test/raw/t1_raw.nii.gz -t1c /home/home/lucas/scripts/test/raw/t1c_raw.nii.gz -t2 /home/home/lucas/scripts/test/raw/t2_raw.nii.gz -flair /home/home/lucas/scripts/test/raw/flair_raw.nii.gz -outdir /home/home/lucas/scripts/test/stripped  
    parser = argparse.ArgumentParser()
    parser.add_argument("-t1", type=str, help="Path to T1 nifti.")
    parser.add_argument("-t1c", type=str, help="Path to T1 contrast nifti.")
    parser.add_argument("-t2", type=str, help="Path to T2 nifti.")
    parser.add_argument("-flair", type=str, help="Path to flair nifti.")
    parser.add_argument("-outdir", type=str, help="Output directory.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]="2"


    run_preprocessing(args.t1, args.t1c, args.t2, args.flair, args.outdir)
