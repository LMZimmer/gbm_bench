import os
import ants
import json
import shutil
import argparse
import numpy as np
import nibabel as nib
from scipy import stats
from pathlib import Path
from gbm_bench.utils.utils import merge_pdfs, load_mri_data
from gbm_bench.utils.constants import *
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.utils.visualization import plot_tumor_sizes, plot_performances, plot_com_distances


def register_and_save(fixed_path, moving_path, coreg_path, outfile, coreg_outfile):
    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))

    print(f"Registering {moving_path} to {fixed_path} and saving to {outfile}...")
    reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="antsRegistrationSyN[s,2]",
            )

    ants.image_write(reg["warpedmovout"], str(outfile))

    print(f"Co-registering {coreg_path} and saving to {coreg_outfile}")
    coreg = ants.image_read(str(coreg_path))
    coreg_warped = ants.apply_transforms(
            fixed=fixed,
            moving=coreg,
            transformlist=reg["fwdtransforms"],
            interpolator='linear'
            )
    ants.image_write(coreg_warped, str(coreg_outfile))


if __name__ == "__main__":
    # Example:
    # python scripts/extract_and_register.py -outdir /mnt/Drive2/lucas/mara_nnunet_data/
    # python -u scripts/extract_and_register.py -outdir /mnt/Drive2/lucas/mara_nnunet_data/ > extract_postop.txt 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-outdir", type=str, help="Algorithm ID to evaluate.")
    args = parser.parse_args()

    DATASET_IDS = ["RHUH", "LUMIERE", "GLIODIL"] #"TCGA-LGG"]
    DATASET_DIRS = [RHUH_GBM_DIR, LUMIERE_DIR, GLIODIL_DIR] #TCGA_LGG_DIR]
    ROOT_DIRS = [
            "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM",
            "/mnt/Drive2/lucas/datasets/LUMIERE/Imaging",
            "/mnt/Drive2/lucas/datasets/GLIODIL",
            #"/mnt/Drive2/lucas/datasets/TCGA-LGG"
            ]

    for d_id, d_dir, d_rootdir in zip(DATASET_IDS, DATASET_DIRS, ROOT_DIRS):
        print(f"Evaluating {d_id}...")

        dataset = LongitudinalDataset(dataset_id=d_id, root_dir=d_rootdir)
        dataset.load(d_dir)

        dataset_results = []
        for patient_ind, patient in enumerate(dataset.patients):
            patient_id = patient["patient_id"]
            preop_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]  # Find first preop exam
            
            postop_exams = dataset.get_patient_exams(patient_id=patient_id, timepoint="postop")
            if len(postop_exams) < 1:
                continue
            postop_exam = postop_exams[0]

            if "UPENN" in d_id:
                preop_exam_dir = preop_exam["t1"].parent
                followup_exam_dir = followup_exam["t1"].parent
            elif "GLIODIL" in d_id:
                no_t1c = [
                "data_001",
                "data_013",
                "data_020",
                "data_030",
                "data_034",
                "data_991",
                "data_992",
                "data_994",
                "data_995",
                "data_998"
                ]
                if patient_id in no_t1c:
                    preop_exam_dir = preop_exam["t1c"].parent / "preop"
                    postop_exam_dir = postop_exam["t1c"].parent / "postop"
                else:
                    preop_exam_dir = preop_exam["t1c"].parent / "preop"
                    postop_exam_dir = postop_exam["t1c"].parent / "postop"
            else:
                preop_exam_dir = preop_exam["t1c"].parent
                postop_exam_dir = postop_exam["t1c"].parent

            try:
                copy_files = [
                        RECURRENCE_SCHEMA.format(base_dir=postop_exam_dir),
                        ]

                patient_outdir = Path(args.outdir) / d_id / patient_id

                for cf in copy_files:
                    if cf.is_file():
                        shutil.copy(str(cf), str(patient_outdir / ("postop_" + cf.name)))
                    else:
                        if "mask" in str(cf):
                            t1c = load_mri_data(str(backup))
                            background = np.min(t1c)
                            brain_mask = np.rint(t1c > background).astype(np.int32)
                            brain_mask_nii = nib.Nifti1Image(brain_mask, np.eye(4))
                            nib.save(brain_mask_nii, str(patient_outdir / cf.name))

                fixed_path = MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_exam_dir, modality="t1c")
                moving_path = MODALITY_STRIPPED_SCHEMA.format(base_dir=postop_exam_dir, modality="t1c")
                coreg_path = MODALITY_STRIPPED_SCHEMA.format(base_dir=postop_exam_dir, modality="flair")

                outfile = str(patient_outdir / ("postop_" + moving_path.name))
                coreg_outfile = str(patient_outdir / ("postop_" + coreg_path.name))

                register_and_save(fixed_path, moving_path, coreg_path, outfile, coreg_outfile)

            except Exception as e:
                print(f"Exception for {patient_id}: {e}")
