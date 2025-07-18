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
    # python scripts/evaluate_datasets.py -algorithm nnUnet > full_nnunet.txt
    parser = argparse.ArgumentParser()
    parser.add_argument("-algorithm", type=str, help="Algorithm ID to evaluate.")
    args = parser.parse_args()

    DATASET_IDS = ["RHUH", "UPENN", "LUMIERE", "GLIODIL", "IVYGAP", "CPTAC", "TCGA-GBM"] #"TCGA-LGG"]
    DATASET_DIRS = [RHUH_GBM_DIR, UPENN_GBM_DIR, LUMIERE_DIR, GLIODIL_DIR, IVYGAP_DIR, CPTAC_DIR, TCGA_GBM_DIR] #TCGA_LGG_DIR]
    ROOT_DIRS = [
            "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM",
            "/home/home/lucas/data/UPENN-GBM/UPENN-GBM",
            "/mnt/Drive2/lucas/datasets/LUMIERE/Imaging",
            "/mnt/Drive2/lucas/datasets/GLIODIL",
            "/mnt/Drive2/lucas/datasets/IVYGAP",
            "/mnt/Drive2/lucas/datasets/CPTAC-GBM",
            "/mnt/Drive2/lucas/datasets/TCGA-GBM",
            #"/mnt/Drive2/lucas/datasets/TCGA-LGG"
            ]

    print(f"Evaluating {args.algorithm}")
    all_results = []

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
                
                performance_dir = METRICS_SCHEMA.format(base_dir=followup_exam_dir, algo_id=args.algorithm)
                performance_dict = json.load(open(performance_dir, "r"))
                
                if "roc_auc_model" in performance_dict.keys():
                    all_results.append(performance_dict)
                    dataset_results.append(performance_dict)
            except Exception as e:
                print(f"Exception for {patient_id}: {e}")
        
        cov_std = [r["recurrence_coverage_standard"] for r in dataset_results]
        cov_std_all = [r["recurrence_coverage_standard_all"] for r in dataset_results]
        cov_mod = [r["recurrence_coverage_model"] for r in dataset_results]
        cov_mod_all = [r["recurrence_coverage_model_all"] for r in dataset_results]
        roc_auc_model_all = [r["roc_auc_model"] for r in dataset_results]
        roc_auc_standard_fade = [r["roc_auc_standard_fade"] for r in dataset_results]
        print(f"{d_id}: {stats.wilcoxon(cov_std, cov_mod, alternative='less')} / {stats.wilcoxon(cov_std_all, cov_mod_all, alternative='less')}")
        print(f"{d_id} (model): {np.mean(roc_auc_model_all):.2f} \u00B1 {stats.sem(roc_auc_model_all):.2f}")
        print(f"{d_id} (fade): {np.mean(roc_auc_standard_fade):.2f} \u00B1 {stats.sem(roc_auc_standard_fade):.2f}")

    recurrence_coverage_standard = [r["recurrence_coverage_standard"] for r in all_results]
    recurrence_coverage_standard_all = [r["recurrence_coverage_standard_all"] for r in all_results]
    recurrence_coverage_model = [r["recurrence_coverage_model"] for r in all_results]
    recurrence_coverage_model_all = [r["recurrence_coverage_model_all"] for r in all_results]
    roc_auc_model_all = [r["roc_auc_model"] for r in all_results]
    roc_auc_standard_fade_all = [r["roc_auc_standard_fade"] for r in all_results]
    missed_standard = [r["missed_voxels_standard"] / 1000 for r in all_results]
    missed_standard_all = [r["missed_voxels_standard_all"] / 1000 for r in all_results]
    missed_model = [r["missed_voxels_model"] / 1000 for r in all_results]
    missed_model_all = [r["missed_voxels_model_all"] / 1000 for r in all_results]
    missed_diff = [ms - mm for ms, mm in zip(missed_standard, missed_model)]

    print(f"Finished evaluation.")
    print(f"Standard plan coverge: {100*np.mean(recurrence_coverage_standard):.2f} \u00B1 {100*stats.sem(recurrence_coverage_standard):.2f}")
    print(f"Standard plan coverge (all): {100*np.mean(recurrence_coverage_standard_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_standard_all):.2f}")
    print(f"Model plan coverge: {100*np.mean(recurrence_coverage_model):.2f} \u00B1 {100*stats.sem(recurrence_coverage_model):.2f}")
    print(f"Model plan coverge (all): {100*np.mean(recurrence_coverage_model_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_model_all):.2f}")
    print(f"Combined: {stats.wilcoxon(recurrence_coverage_standard, recurrence_coverage_model, alternative='less')} / {stats.wilcoxon(recurrence_coverage_standard_all, recurrence_coverage_model_all, alternative='less')}")
    
    print(f"\n")
    print(f"ROC AUC Model: {np.mean(roc_auc_model_all):.2f} \u00B1 {stats.sem(roc_auc_model_all):.2f}")
    print(f"ROC AUC Std plan: {np.mean(roc_auc_standard_fade):.2f} \u00B1 {stats.sem(roc_auc_standard_fade):.2f}")

    print(f"\n")
    print(f"Standard plan missed: {np.mean(missed_standard):.2f} \u00B1 {stats.sem(missed_standard):.2f}")
    print(f"Standard plan missed (all): {np.mean(missed_standard_all):.2f} \u00B1 {stats.sem(missed_standard_all):.2f}")
    print(f"Model plan missed: {np.mean(missed_model):.2f} \u00B1 {stats.sem(missed_model):.2f}")
    print(f"Model plan missed (all): {np.mean(missed_model_all):.2f} \u00B1 {stats.sem(missed_model_all):.2f}")
    print(f"Difference: {np.mean(missed_diff):.2f} \u00B1 {stats.sem(missed_diff):.2f}")
    print(f"Missed: {stats.wilcoxon(missed_model, missed_standard, alternative='less')} / {stats.wilcoxon(missed_model_all, missed_standard_all, alternative='less')}")
