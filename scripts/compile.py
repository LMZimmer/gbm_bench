import os
import json
import shutil
import argparse
import numpy as np
from scipy import stats
from pathlib import Path
from gbm_bench.utils.utils import merge_pdfs
from gbm_bench.utils.constants import *
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.utils.visualization import plot_tumor_sizes, plot_performances, plot_com_distances


MODALITIES = ["t1c", "flair", "t1", "t2"]


def copy_file(source, dest_base_dir):
    new_file = dest_base_dir / source.name
    shutil.copy(source, str(new_file))


if __name__ == "__main__":
    # Example:
    # python scripts/compile.py

    DATASET_IDS = ["UPENN", "IVYGAP", "CPTAC", "TCGA-GBM"] #"TCGA-LGG"]
    DATASET_DIRS = [UPENN_GBM_DIR, IVYGAP_DIR, CPTAC_DIR, TCGA_GBM_DIR] #TCGA_LGG_DIR]
    ROOT_DIRS = [
            "/home/home/lucas/data/UPENN-GBM/UPENN-GBM",
            "/mnt/Drive2/lucas/datasets/IVYGAP",
            "/mnt/Drive2/lucas/datasets/CPTAC-GBM",
            "/mnt/Drive2/lucas/datasets/TCGA-GBM",
            #"/mnt/Drive2/lucas/datasets/TCGA-LGG"
            ]

    outdir = Path("/mnt/Drive2/lucas/datasets/4_datasets_recurrence_check")
    outdir.mkdir(exist_ok=True, parents=True)

    for d_id, d_dir, d_rootdir in zip(DATASET_IDS, DATASET_DIRS, ROOT_DIRS):
        print(f"Evaluating {d_id}...")

        dataset = LongitudinalDataset(dataset_id=d_id, root_dir=d_rootdir)
        dataset.load(d_dir)

        dataset_outdir = outdir / d_id
        dataset_outdir.mkdir(exist_ok=True)

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

            outdir_preop = dataset_outdir / patient_id / "preop"
            outdir_followup = dataset_outdir / patient_id / "followup"
            outdir_preop.mkdir(exist_ok=True, parents=True)
            outdir_followup.mkdir(exist_ok=True, parents=True)

            tumorseg_file = TUMORSEG_SCHEMA.format(base_dir=preop_exam_dir)
            recurrenceseg_file = TUMORSEG_SCHEMA.format(base_dir=followup_exam_dir)

            modalities_preop = [MODALITY_CONVERTED_SCHEMA.format(base_dir=preop_exam_dir, modality=m) for m in MODALITIES]
            modalities_followup = [MODALITY_CONVERTED_SCHEMA.format(base_dir=followup_exam_dir, modality=m) for m in MODALITIES]

            stripped_preop = [MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_exam_dir, modality=m) for m in MODALITIES]
            stripped_followup = [MODALITY_STRIPPED_SCHEMA.format(base_dir=followup_exam_dir, modality=m) for m in MODALITIES]

            copy_file(tumorseg_file, outdir_preop)
            copy_file(recurrenceseg_file, outdir_followup)

            for m1, m2, m3, m4 in zip(modalities_preop, stripped_preop, modalities_followup, stripped_followup):
                copy_file(m1, outdir_preop)
                copy_file(m2, outdir_preop)
                copy_file(m3, outdir_followup)
                copy_file(m4, outdir_followup)
