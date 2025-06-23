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


if __name__ == "__main__":
    # Example:
    # python scripts/evaluate_datasets.py -algorithm sbtc > full_sbtc.txt
    # python scripts/evaluate_datasets.py -algorithm gliodil > full_gliodil.txt
    # python scripts/evaluate_datasets.py -algorithm lmi > full_lmi.txt
    parser = argparse.ArgumentParser()
    parser.add_argument("-algorithm", type=str, help="Algorithm ID to evaluate.")
    args = parser.parse_args()

    DATASET_IDS = ["RHUH", "UPENN", "LUMIERE", "GLIODIL", "IVYGAP", "CPTAC", "TCGA-GBM", "TCGA-LGG"]
    DATASET_DIRS = [RHUH_GBM_DIR, UPENN_GBM_DIR, LUMIERE_DIR, GLIODIL_DIR, IVYGAP_DIR, CPTAC_DIR, TCGA_GBM_DIR, TCGA_LGG_DIR]
    ROOT_DIRS = [
            "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM",
            "/home/home/lucas/data/UPENN-GBM/UPENN-GBM",
            "/mnt/Drive2/lucas/datasets/LUMIERE/Imaging",
            "/mnt/Drive2/lucas/datasets/GLIODIL",
            "/mnt/Drive2/lucas/datasets/IVYGAP",
            "/mnt/Drive2/lucas/datasets/CPTAC-GBM",
            "/mnt/Drive2/lucas/datasets/TCGA-GBM",
            "/mnt/Drive2/lucas/datasets/TCGA-LGG"
            ]

    print(f"Evaluating {args.algorithm}")
    all_results = []

    for d_id, d_dir, d_rootdir in zip(DATASET_IDS, DATASET_DIRS, ROOT_DIRS):
        print(f"Evaluating {d_id}...")

        dataset = LongitudinalDataset(dataset_id=d_id, root_dir=d_rootdir)
        dataset.load(d_dir)

        for patient_ind, patient in enumerate(dataset.patients):
            patient_id = patient["patient_id"]
            preop_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]  # Find first preop exam
            followup_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="followup")[0]

            if "UPENN" in d_id:
                preop_exam_dir = preop_exam["t1"].parent
                followup_exam_dir = followup_exam["t1"].parent
            elif "GLIODIL" in d_id:
                preop_exam_dir = preop_exam["t1c"].parent / "preop"
                followup_exam_dir = followup_exam["t1c"].parent / "followup"
            else:
                preop_exam_dir = followup_exam["t1c"].parent
                followup_exam_dir = followup_exam["t1c"].parent

            try:
                performance_dir = METRICS_SCHEMA.format(base_dir=followup_exam_dir, algo_id=args.algorithm.lower())
                performance_dict = json.load(open(performance_dir, "r"))
                all_results.append(performance_dict)
            except Exception as e:
                print(f"Exception for {patient_id}: {e}")

    recurrence_coverage_standard = [r["recurrence_coverage_standard"] for r in all_results]
    recurrence_coverage_standard_all = [r["recurrence_coverage_standard_all"] for r in all_results]
    recurrence_coverage_model = [r["recurrence_coverage_model"] for r in all_results]
    recurrence_coverage_model_all = [r["recurrence_coverage_model_all"] for r in all_results]

    print(f"Finished evaluation.")
    print(f"Standard plan coverge: {100*np.mean(recurrence_coverage_standard):.2f} \u00B1 {100*stats.sem(recurrence_coverage_standard):.2f}")
    print(f"Standard plan coverge (all): {100*np.mean(recurrence_coverage_standard_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_standard_all):.2f}")
    print(f"Model plan coverge: {100*np.mean(recurrence_coverage_model):.2f} \u00B1 {100*stats.sem(recurrence_coverage_model):.2f}")
    print(f"Model plan coverge (all): {100*np.mean(recurrence_coverage_model_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_model_all):.2f}")

