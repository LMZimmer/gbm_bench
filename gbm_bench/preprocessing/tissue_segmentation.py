import os
import sys
import glob
import ants
import argparse
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from loguru import logger
from brainles_preprocessing.preprocessor import Preprocessor
from brainles_preprocessing.registration import ANTsRegistrator
from brainles_preprocessing.modality import Modality, CenterModality
from brainles_preprocessing.normalization.percentile_normalizer import PercentileNormalizer
from gbm_bench.utils.constants import (
    ATLAS_DIR,
    HEALTHY_BRAIN_MASK_SCHEMA,
    TISSUE_PBMAP_SCHEMA,
    TISSUE_SCHEMA,
    TISSUE_SEG_SCHEMA,
    TISSUE_SEG_BASE_SCHEMA
)


def generate_healthy_brain_mask(brain_mask_file: Path, tumor_seg_file: Path, outfile: Path) -> None:
    """
    Generate a healthy brain mask by subtracting the tumor segmentation from the brain mask.

    Parameters:
        brain_mask_file (Path): Path to the brain mask NIfTI file.
        tumor_seg_file (Path): Path to the tumor segmentation NIfTI file.
        outfile (Path): Output file path where the healthy brain mask will be saved.

    Returns:
        None
    """
    logger.debug("Generating healthy brain mask.")
    # Load niftis
    brain_nifti = nib.load(str(brain_mask_file))
    affine, header = brain_nifti.affine, brain_nifti.header
    brain_data = brain_nifti.get_fdata()

    tumor_data = nib.load(str(tumor_seg_file)).get_fdata()
    tumor_mask = (tumor_data > 0).astype(np.float32)

    # Generate the healthy brain mask.
    healthy_data = np.where(tumor_mask > 0, 0, brain_data).astype(np.float32)

    # Generate output nifti and save it
    outfile.parent.mkdir(parents=True, exist_ok=True)
    healthy_mask_nifti = nib.Nifti1Image(healthy_data, affine, header)
    nib.save(healthy_mask_nifti, str(outfile))
    
    logger.debug(f"Healthy brain mask generated succesfully and saved to {outfile}")


def run_tissue_seg_registration(t1_file: Path, healthy_mask_file: Path, outdir: Path, brain_mask_file: Path = None, refit_brain: bool = False) -> None:
    """
    Performs tissue segmentation for gm, wm, csf by registering an atlas to the input t1 file and transforming atlas tissue maps using
    the obtained transformation.

    Parameters:
        t1_file (Path): Path to the t1 nifti.
        healthy_mask_file (Path): Path to the healthy brain mask nifti. 
        outdir (Path): Path to output directory. Usually exam directory.
        brain_mask_file (Path): Path to the brain mask nifti as obtained from skull stripping.
        refit_brain (bool): Wether to refit the outline of the warped atlas to the brain mask.

    Returns:
        None
    """
    logger.info(f"Starting tissue segmentation.")
    # Prepare directories
    atlas_base_dir = ATLAS_DIR
    atlas_t1_dir = ATLAS_DIR / "t1.nii"
    atlas_tissues_dir = ATLAS_DIR / "tissues.nii"
    atlas_pbmap_dirs = {tissue: ATLAS_DIR / f"pbmap_{tissue}.nii" for tissue in ["csf", "gm", "wm"]}
    
    outprefix = TISSUE_SEG_BASE_SCHEMA.format(outdir=outdir)
    outprefix.mkdir(parents=True, exist_ok=True)

    # Read images
    healthy_mask = ants.image_read(str(healthy_mask_file))

    t1_patient = ants.image_read(str(t1_file))
    t1_atlas = ants.image_read(str(atlas_t1_dir))
    
    # Register atlas to patient deformably
    reg = ants.registration(
            fixed=t1_patient,
            moving=t1_atlas,
            type_of_transform="antsRegistrationSyN[s,2]",
            mask=healthy_mask,
            outprefix=str(outprefix)+"/"
            )
    transforms_path = reg['fwdtransforms']

    # Transform atlas tissues deformably
    tissues_atlas = ants.image_read(str(atlas_tissues_dir))
    tissues_warped = ants.apply_transforms(
            fixed=t1_patient,
            moving=tissues_atlas, 
            transformlist=transforms_path,
            interpolator="nearestNeighbor"
            )

    """
    # Refit tissue mask on the full brain mask, if desired
    if refit_brain:
        if brain_mask_file is None:
            raise ValueError(f"Please specify brain_mask_file when using refit_brain=True")
        
        logger.info(f"refit_brain set to True. Refitting to the brain mask.")
        brain_mask = ants.image_read(str(brain_mask_file))
        tissue_mask_nib =  nib.Nifti1Image(
                (tissues_warped.numpy() > 0.5).astype(np.int32),
                header=tissues_warped.to_nibabel().header,
                affine=tissues_warped.to_nibabel().affine
                )
        tissue_mask = ants.from_nibabel(tissue_mask_nib)

        reg2 = ants.registration(
                fixed=brain_mask,
                moving=tissue_mask,
                type_of_transform="antsRegistrationSyN[bo]",
                outprefix=str(outprefix)+"/"
                )
        transforms_path_masks = reg2['fwdtransforms']

        tissues_warped = ants.apply_transforms(
                fixed=brain_mask,
                moving=tissues_warped,
                transformlist=transforms_path_masks,
                interpolator="nearestNeighbor"
                )
    """

    # Save transformed tissue segmentation
    tissues_warped_nifti = tissues_warped.to_nibabel()
    nib.save(tissues_warped_nifti, str(TISSUE_SEG_SCHEMA.format(outdir=outdir)))

    logger.debug(f"Registration step done, saving output to {TISSUE_SEG_SCHEMA.format(outdir=outdir)}")
    logger.info(f"Generating pbmaps...")

    # Create single tissue masks from full tissue segmentation
    tissue_labels = {"csf": 1., "gm": 2., "wm": 3.}
    header, aff = tissues_warped_nifti.header, tissues_warped_nifti.affine
    for tissue, label in tissue_labels.items():
        tissue_mask = (tissues_warped.numpy() == label).astype(np.int32)
        tissue_mask_nifti = nib.Nifti1Image(tissue_mask, header=header, affine=aff)
        nib.save(tissue_mask_nifti, str(TISSUE_SCHEMA.format(outdir=outdir, tissue=tissue)))

    # Create probability maps by transforming atlas pbmaps with the previously obtained transformation
    for tissue, pbmap_dir in atlas_pbmap_dirs.items():
        pbmap = ants.image_read(str(pbmap_dir))
        warped_pbmap = ants.apply_transforms(
                fixed=t1_patient,
                moving=pbmap,
                transformlist=transforms_path,
                interpolator="linear"
                )

        """
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
        """

        warped_pbmap_nifti = warped_pbmap.to_nibabel()
        nib.save(warped_pbmap_nifti, str(TISSUE_PBMAP_SCHEMA.format(outdir=outdir, tissue=tissue)))
    
    logger.info(f"Tissue segmentation finished succesfully.")


if __name__ == "__main__":
    # Example:
    # python gbm_bench/preprocessing/tissue_segmentation.py -cuda_device 4
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    outdir = Path("test_data/exam1/preprocessing/")
    t1c_file = outdir / "skull_stripped/t1c_bet_normalized.nii.gz"
    brain_mask_file = outdir / "skull_stripped/t1c_bet_mask.nii.gz"
    tumor_seg_file = outdir / "tumor_segmentation/tumor_seg.nii.gz"

    outdir = Path("./tmp_test_tissueseg")

    healthy_mask_file = HEALTHY_BRAIN_MASK_SCHEMA.format(outdir=outdir)
    generate_healthy_brain_mask(
            brain_mask_file=brain_mask_file,
            tumor_seg_file=tumor_seg_file,
            outfile=healthy_mask_file
            )

    run_tissue_seg_registration(
            t1_file=t1c_file,
            healthy_mask_file=healthy_mask_file,
            brain_mask_file=brain_mask_file,
            outdir=outdir,
            refit_brain=False
            )
