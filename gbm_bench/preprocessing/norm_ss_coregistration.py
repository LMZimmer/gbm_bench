import os
import ants
import shutil
import argparse
from brainles_preprocessing.preprocessor import Preprocessor
from brainles_preprocessing.modality import Modality, CenterModality
from brainles_preprocessing.normalization.percentile_normalizer import PercentileNormalizer


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


def norm_ss_coregister(t1, t1c, t2, flair, outdir):

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


def register_longitudinal(t1c_pre_dir, t1c_post_dir, outdir, transform_images_dict, transform_masks_dict):
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
    #ants.write_transform(reg["warpedmovout"], os.path.join(outdir, "longitudinal_trafo.mat"))
    shutil.copyfile(src=reg["fwdtransforms"][0], dst=os.path.join(outdir, "longitudinal_trafo.mat"))
    ants.image_write(reg["warpedmovout"], os.path.join(outdir, "t1c_warped_longitudinal.nii.gz"))

    for img_id, img_dir in transform_images_dict.items():
        img = ants.image_read(img_dir)
        img_warped = ants.apply_transforms(
                fixed=t1c_pre,
                moving=img,
                transformlist=reg["fwdtransforms"],
                interpolator='linear'
                )
        ants.image_write(img_warped, os.path.join(outdir, f"{img_id}_warped_longitudinal.nii.gz"))

    for mask_id, mask_dir in transform_masks_dict.items():
        mask = ants.image_read(mask_dir)
        mask_warped = ants.apply_transforms(
                fixed=t1c_pre,
                moving=mask,
                transformlist=reg["fwdtransforms"],
                interpolator='nearestNeighbor'
                )
        ants.image_write(mask_warped, os.path.join(outdir, f"{img_id}_warped_longitudinal.nii.gz"))


if __name__ == "__main__":
    # python gbm_bench/preprocessing/norm_ss_coregistration.py
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    t1c_pre_dir = "/home/home/lucas/projects/gbm_bench/test_data/exam1/preprocessing/skull_stripped/t1c_bet_normalized.nii.gz"
    t1c_post_dir = "/home/home/lucas/projects/gbm_bench/test_data/exam2/preprocessing/skull_stripped/t1c_bet_normalized.nii.gz"
    outdir = "/home/home/lucas/projects/gbm_bench/test_data/exam2/preprocessing/longitudinal/"
    transform_images_dict = {
            "t1": "/home/home/lucas/projects/gbm_bench/test_data/exam2/preprocessing/skull_stripped/t1_bet_normalized.nii.gz",
            "t2": "/home/home/lucas/projects/gbm_bench/test_data/exam2/preprocessing/skull_stripped/t2_bet_normalized.nii.gz",
            "flair": "/home/home/lucas/projects/gbm_bench/test_data/exam2/preprocessing/skull_stripped/flair_bet_normalized.nii.gz"
            }
    transform_masks_dict = {
            "tumor_seg": "/home/home/lucas/projects/gbm_bench/test_data/exam2/preprocessing/tumor_segmentation/tumor_seg.nii.gz"
            }
    register_longitudinal(
            t1c_pre_dir=t1c_pre_dir,
            t1c_post_dir=t1c_post_dir,
            outdir=outdir,
            transform_images_dict=transform_images_dict,
            transform_masks_dict=transform_masks_dict
            )

    """
    # Example:
    # python gbm_bench/preprocessing/norm_ss_coregistration.py -t1 test_data/exam1/preprocessing/nifti_conversion/t1.nii.gz -t1c test_data/exam1/preprocessing/nifti_conversion/t1c.nii.gz -t2 test_data/exam1/preprocessing/nifti_conversion/t2.nii.gz -flair test_data/exam1/preprocessing/nifti_conversion/flair.nii.gz -outdir tmp_test_ss/
    parser = argparse.ArgumentParser()
    parser.add_argument("-t1", type=str, help="Path to T1 nifti.")
    parser.add_argument("-t1c", type=str, help="Path to T1 contrast nifti.")
    parser.add_argument("-t2", type=str, help="Path to T2 nifti.")
    parser.add_argument("-flair", type=str, help="Path to flair nifti.")
    parser.add_argument("-outdir", type=str, help="Output directory.")
    parser.add_argument("-cuda_device", type=str, default="1", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    norm_ss_coregister(
            t1=args.t1,
            t1c=args.t1c,
            t2=args.t2,
            flair=args.flair,
            outdir=args.outdir
            )

    # Testing register_longitudinal
    t1c_pre_dir = "/home/home/lucas/projects/gbm_bench/test_data/exam1/preprocessing/skull_stripped/t1c_bet_normalized.nii.gz"
    t1c_post_dir = "/home/home/lucas/projects/gbm_bench/test_data/exam2/preprocessing/skull_stripped/t1c_bet_normalized.nii.gz"
    outdir = "/home/home/lucas/projects/gbm_bench/test_data/exam2/preprocessing/longitudinal/"
    transform_images_dict = {
            "t1": "/home/home/lucas/projects/gbm_bench/test_data/exam2/preprocessing/skull_stripped/t1_bet_normalized.nii.gz",
            "t2": "/home/home/lucas/projects/gbm_bench/test_data/exam2/preprocessing/skull_stripped/t2_bet_normalized.nii.gz",
            "flair": "/home/home/lucas/projects/gbm_bench/test_data/exam2/preprocessing/skull_stripped/flair_bet_normalized.nii.gz"
            }
    transform_masks_dict = {
            "tumor_seg": "/home/home/lucas/projects/gbm_bench/test_data/exam2/preprocessing/tumor_segmentation/tumor_seg.nii.gz"
            }
    register_longitudinal(
            t1c_pre_dir=t1c_pre_dir,
            t1c_post_dir=t1c_post_dir,
            outdir=outdir,
            transform_images_dict=transform_images_dict,
            transform_masks_dict=transform_masks_dict
            )
    """
