import argparse
import ants
import numpy as np
import nibabel as nib
import os
import subprocess
import glob
import nibabel as nib
from pathlib import Path
from brainles_preprocessing.modality import Modality, CenterModality
from brainles_preprocessing.normalization.percentile_normalizer import (
    PercentileNormalizer,
)
from brainles_preprocessing.preprocessor import Preprocessor
from brainles_preprocessing.registration import ANTsRegistrator
from typing import List, Optional
import matplotlib.pyplot as plt
import sys
import ants


def generate_healthy_brain_mask(brain_mask_file, tumor_mask_file, outdir):
    #TODO: better implementation via a[b > 0] = 0
    brain_nifti = nib.load(brain_mask_file)
    aff, header = brain_nifti.affine, brain_nifti.header
    m1 = brain_nifti.get_fdata()
    m2 = (nib.load(tumor_mask_file).get_fdata() > 0).astype(np.float32)
    healthy_mask = ((m1 - m2) > 0).astype(np.float32)
    healthy_mask_nifti = nib.Nifti1Image(healthy_mask, aff, header)
    nib.save(healthy_mask_nifti, outdir)


def run_tissue_seg_registration(t1_file, outdir, healthy_mask_dir, brain_mask_dir=None, refit_brain=False):
    # TODO: Make the atlas path an argument
    sri_42_atlas = "/home/home/lucas/bin/miniconda3/envs/brainles/lib/python3.10/site-packages/brainles_preprocessing/registration/atlas/t1_skullstripped_brats_space.nii"
    sri_42_tissues = "/home/home/lucas/data/ATLAS/SRI-24/tissues.nii"

    matrix_dir = os.path.join(outdir, "affine.mat")
    log_dir = os.path.join(outdir, "tissue_affine_reg.log")
    healthy_mask = ants.image_read(healthy_mask_dir)

    os.makedirs(outdir, exist_ok=True)
    t1_patient = ants.image_read(t1_file)
    t1_atlas = ants.image_read(sri_42_atlas)
    
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
    tissues_atlas = ants.image_read(sri_42_tissues)
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
        tissue_mask = ants.get_mask(warped_tissues, low_thresh=0.5)
        reg2 = ants.registration(
                fixed=brain_mask,
                moving=tissue_mask,
                type_of_transform="antsRegistrationSyN[s,2]",
                outprefix=os.path.join(outdir, '')
                )
        transforms_path_masks = reg['fwdtransforms']

        warped_tissues = ants.apply_transforms(
                fixed=t1_patient,
                moving=warped_tissues,
                transformlist=transforms_path_masks,
                interpolator="nearestNeighbor"
                )

    warped_tissues_nifti = warped_tissues.to_nibabel()
    nib.save(warped_tissues_nifti, os.path.join(outdir, "tissue_seg.nii.gz"))

    # 1:csf, 2:gm, 3:wm
    header, aff = warped_tissues_nifti.header, warped_tissues_nifti.affine
    csf = (warped_tissues.numpy() == 1.).astype(np.int32)
    gm = (warped_tissues.numpy() == 2.).astype(np.int32)
    wm = (warped_tissues.numpy() == 3.).astype(np.int32)

    csf_nifti = nib.Nifti1Image(csf, header=header, affine=aff)
    gm_nifti = nib.Nifti1Image(gm, header=header, affine=aff)
    wm_nifti = nib.Nifti1Image(wm, header=header, affine=aff)

    nib.save(csf_nifti, os.path.join(outdir, "csf.nii.gz"))
    nib.save(gm_nifti, os.path.join(outdir, "gm.nii.gz"))
    nib.save(wm_nifti, os.path.join(outdir, "wm.nii.gz"))


if __name__ == "__main__":
    # python 4_tissue_segmentation.py -t1 /home/home/lucas/scripts/test/stripped/t1c_bet_normalized.nii.gz -brain_mask /home/home/lucas/scripts/test/stripped/t1c_bet_mask.nii.gz -tumor_mask /home/home/lucas/scripts/test/segmentations/brats_segmentation.nii.gz -outdir /home/home/lucas/scripts/test/segmentations
    # python 4_tissue_segmentation.py -t1 /home/home/lucas/data/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0001/0/RHUH-0001_0_t1.nii.gz -brain_mask /home/home/lucas/inverse_tumor_mask.nii.gz -tumor_mask /home/home/lucas/data/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0001/0/RHUH-0001_0_segmentations.nii.gz -outdir /home/home/lucas/scripts/test/segmentations
    parser = argparse.ArgumentParser()
    parser.add_argument("-t1", type=str, help="Path to T1 nifti.")
    parser.add_argument("-brain_mask", type=str, help="Path to brain mask.")
    parser.add_argument("-tumor_mask", type=str, help="Path to tumor mask.")
    parser.add_argument("-outdir", type=str, help="Desired file path for output segmentation.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"


    print("Generating healthy brain mask...")
    healthy_mask_dir = os.path.join(args.outdir, "healthy_brain_mask.nii.gz")
    generate_healthy_brain_mask(args.brain_mask, args.tumor_mask, healthy_mask_dir)

    run_tissue_seg_registration(t1_file=args.t1, outdir=args.outdir, healthy_mask_dir=healthy_mask_dir, brain_mask_dir=args.brain_mask)
