import os
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


if __name__ == "__main__":
    # Example:
    # python scripts/extract_dirac.py -outdir /mnt/Drive2/lucas/models/DIRAC/Dataset/predict_gbm
    parser = argparse.ArgumentParser()
    parser.add_argument("-outdir", type=str, help="Algorithm ID to evaluate.")
    args = parser.parse_args()

    DATASET_IDS = ["RHUH", "LUMIERE", "GLIODIL"] #"TCGA-LGG"]
    DATASET_DIRS = [RHUH_GBM_DIR, LUMIERE_DIR, GLIODIL_DIR] #TCGA_LGG_DIR]
    ROOT_DIRS = [
            "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM",
            "/mnt/Drive2/lucas/datasets/LUMIERE/Imaging",
            "/mnt/Drive2/lucas/datasets/GLIODIL",
            ]

    for d_id, d_dir, d_rootdir in zip(DATASET_IDS, DATASET_DIRS, ROOT_DIRS):
        print(f"Evaluating {d_id}...")

        dataset = LongitudinalDataset(dataset_id=d_id, root_dir=d_rootdir)
        dataset.load(d_dir)

        dataset_results = []
        for patient_ind, patient in enumerate(dataset.patients):
            patient_id = patient["patient_id"]
            preop_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]  # Find first preop exam
            followup_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="followup")[0]

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
                    preop_exam_dir = preop_exam["tumorseg"].parent / "preop"
                    followup_exam_dir = followup_exam["tumorseg"].parent / "followup"
                else:
                    preop_exam_dir = preop_exam["t1c"].parent / "preop"
                    followup_exam_dir = followup_exam["t1c"].parent / "followup"
            else:
                preop_exam_dir = preop_exam["t1c"].parent
                followup_exam_dir = followup_exam["t1c"].parent

            try:
                copy_files = [
                        TUMORSEG_SCHEMA.format(base_dir=followup_exam_dir),
                        MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_exam_dir, modality="t1c"),
                        #MODALITY_STRIPPED_SCHEMA.format(base_dir=followup_exam_dir, modality="t1c"),
                        MODALITY_STRIPPED_SCHEMA.format(base_dir=followup_exam_dir, modality="flair"),
                        ]
                backup = MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_exam_dir, modality="t1c")

                #patient_outdir = Path(args.outdir) / d_id / patient_id
                patient_outdir = Path(args.outdir) / patient_id
                patient_outdir.mkdir(parents=True, exist_ok=True)

                for cf in copy_files:
                    if cf.is_file():
                        shutil.copy(str(cf), str(patient_outdir / cf.name))
                    else:
                        if "mask" in str(cf):
                            t1c = load_mri_data(str(backup))
                            background = np.min(t1c)
                            brain_mask = np.rint(t1c > background).astype(np.int32)
                            brain_mask_nii = nib.Nifti1Image(brain_mask, np.eye(4))
                            nib.save(brain_mask_nii, str(patient_outdir / cf.name))

            except Exception as e:
                shutil.rmtree(patient_outdir)
                print(f"Exception for {patient_id}: {e}")
