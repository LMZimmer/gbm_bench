import os
import sys
import glob
import ants
import argparse
import subprocess
import numpy as np
import nibabel as nib
from brainles_preprocessing.preprocessor import Preprocessor
from brainles_preprocessing.registration import ANTsRegistrator
from brainles_preprocessing.modality import Modality, CenterModality
from brainles_preprocessing.normalization.percentile_normalizer import PercentileNormalizer


def generate_healthy_brain_mask(brain_mask_file: str, tumor_seg_file: str, outdir: str) -> None:
    #TODO: better implementation via a[b > 0] = 0
    brain_nifti = nib.load(brain_mask_file)
    aff, header = brain_nifti.affine, brain_nifti.header
    m1 = brain_nifti.get_fdata()
    m2 = (nib.load(tumor_mask_file).get_fdata() > 0).astype(np.float32)
    healthy_mask = ((m1 - m2) > 0).astype(np.float32)
    healthy_mask_nifti = nib.Nifti1Image(healthy_mask, aff, header)
    os.makedirs(os.path.dirname(outdir), exist_ok=True)
    nib.save(healthy_mask_nifti, outdir)


def run_tissue_seg_registration(t1_file: str, healthy_mask_dir: str, outdir: str, brain_mask_dir: str = None, refit_brain: bool = False) -> None:

    #TODO absolute paths
    atlas_base_dir = "/home/home/lucas/projects/gbm_bench/gbm_bench/sri24_atlas"
    atlas_t1_dir = os.path.join(atlas_base_dir, "t1.nii")
    atlas_tissues_dir = os.path.join(atlas_base_dir, "tissues.nii")
    atlas_pbmap_dirs = {tissue: os.path.join(atlas_base_dir, f"pbmap_{tissue.upper()}.nii") for tissue in ["csf", "gm", "wm"]}

    matrix_dir = os.path.join(outdir, "affine.mat")
    log_dir = os.path.join(outdir, "tissue_affine_reg.log")
    healthy_mask = ants.image_read(healthy_mask_dir)

    os.makedirs(outdir, exist_ok=True)
    t1_patient = ants.image_read(t1_file)
    t1_atlas = ants.image_read(atlas_t1_dir)
    
    # Register atlas to patient deformably
    reg = ants.registration(
            fixed=t1_patient,
            moving=t1_atlas,
            type_of_transform="antsRegistrationSyN[s,2]",
            mask=healthy_mask,
            outprefix=os.path.join(outdir, '')
            )
    transforms_path = reg['fwdtransforms']

    # Transform atlas tissues deformably
    tissues_atlas = ants.image_read(atlas_tissues_dir)
    warped_tissues = ants.apply_transforms(
            fixed=t1_patient,
            moving=tissues_atlas, 
            transformlist=transforms_path,
            interpolator="nearestNeighbor"
            )

    # Refit tissue mask on the full brain mask, if desired
    if refit_brain:
        if brain_mask_dir is None:
            raise ValueError(f"Please specify brain_maks_dir when using refit_brain=True")
        brain_mask = ants.image_read(brain_mask_dir)
        tissue_mask_nib =  nib.Nifti1Image(
                (warped_tissues.numpy() > 0.5).astype(np.int32),
                header=warped_tissues.to_nibabel().header,
                affine=warped_tissues.to_nibabel().affine
                )
        tissue_mask = ants.from_nibabel(tissue_mask_nib)

        reg2 = ants.registration(
                fixed=brain_mask,
                moving=tissue_mask,
                type_of_transform="antsRegistrationSyN[bo]",
                outprefix=os.path.join(outdir, '')
                )
        transforms_path_masks = reg2['fwdtransforms']

        warped_tissues = ants.apply_transforms(
                fixed=brain_mask,
                moving=warped_tissues,
                transformlist=transforms_path_masks,
                interpolator="nearestNeighbor"
                )

    warped_tissues_nifti = warped_tissues.to_nibabel()
    nib.save(warped_tissues_nifti, os.path.join(outdir, "tissue_seg.nii.gz"))

    # Tissue masks
    tissue_labels = {"csf": 1., "gm": 2., "wm": 3.}
    header, aff = warped_tissues_nifti.header, warped_tissues_nifti.affine
    for tissue, label in tissue_labels.items():
        tissue_mask = (warped_tissues.numpy() == label).astype(np.int32)
        tissue_mask_nifti = nib.Nifti1Image(tissue_mask, header=header, affine=aff)
        nib.save(tissue_mask_nifti, os.path.join(outdir, f"{tissue}.nii.gz"))

    # Probability maps
    for tissue, pbmap_dir in atlas_pbmap_dirs.items():
        pbmap = ants.image_read(pbmap_dir)
        warped_pbmap = ants.apply_transforms(
                fixed=t1_patient,
                moving=pbmap,
                transformlist=transforms_path,
                interpolator="linear"
                )

        if refit_brain:
            #warped_pbmap2 = ants.apply_transforms(
            #        fixed=brain_mask,
            #        moving=warped_pbmap,
            #        transformlist=transforms_path_masks,
            #        interpolator="linear"
            #        )
            #warped_pbmap_nifti2 = warped_pbmap2.to_nibabel()
            #nib.save(warped_pbmap_nifti2, os.path.join(outdir, f"{tissue}_pbmap2.nii.gz"))
            pass

        warped_pbmap_nifti = warped_pbmap.to_nibabel()
        nib.save(warped_pbmap_nifti, os.path.join(outdir, f"{tissue}_pbmap.nii.gz"))


if __name__ == "__main__":
    # Example:
    # python gbm_bench/preprocessing/tissue_segmentation.py -cuda_device 4
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="4", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    t1 = "test_data/exam1/preprocessing/skull_stripped/t1c_bet_normalized.nii.gz"
    brain_mask_dir = "test_data/exam1/preprocessing/skull_stripped/t1c_bet_mask.nii.gz"
    tumor_seg_dir = "test_data/exam1/preprocessing/tumor_segmentation/tumor_seg.nii.gz"

    print("Generating healthy brain mask...")
    healthy_mask_dir = os.path.join(args.outdir, "healthy_brain_mask.nii.gz")
    generate_healthy_brain_mask(
            brain_mask_file=brain_mask_dir,
            tumor_seg_file=tumor_seg,
            outdir=healthy_mask_dir
            )

    print("Starting tissue registration...")
    run_tissue_seg_registration(
            t1_file=t1,
            healthy_mask_dir=healthy_mask_dir,
            brain_mask_dir=brain_mask_dir,
            outdir="./tmp_test_tissueseg",
            refit_brain=False
            )
    print("Finished.")
