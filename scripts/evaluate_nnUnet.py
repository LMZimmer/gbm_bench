import os
import json
import shutil
import argparse
import numpy as np
from scipy import stats
from pathlib import Path
from gbm_bench.utils.constants import *
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.evaluation.evaluate import evaluate_tumor_model


if __name__ == "__main__":
    # Example:
    # python scripts/evaluate_nnUnet.py -dataset rhuh
    # nohup python -u scripts/evaluate_nnUnet.py -dataset rhuh > rhuh_nnUnet.txt 2>&1 &
    # nohup python -u scripts/evaluate_nnUnet.py -dataset upenn > upenn_nnUnet.txt 2>&1 &
    # nohup python -u scripts/evaluate_nnUnet.py -dataset lumiere > lumiere_nnUnet.txt 2>&1 &
    # nohup python -u scripts/evaluate_nnUnet.py -dataset gliodil > gliodil_nnUnet.txt 2>&1 &
    # nohup python -u scripts/evaluate_nnUnet.py -dataset ivygap > ivygap_nnUnet.txt 2>&1 &
    # nohup python -u scripts/evaluate_nnUnet.py -dataset cptac > cptac_nnUnet.txt 2>&1 &
    # nohup python -u scripts/evaluate_nnUnet.py -dataset tcga-gbm > tcga_gbm_nnUnet.txt 2>&1 &
    # nohup python -u scripts/evaluate_nnUnet.py -dataset tcga-lgg > tcga_lgg_nnUnet.txt 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, help="Dataset to evaluate. 'all' for all available datasets.")
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

    result_dirs = [
            "/mnt/Drive2/lucas/models/nnUnet/nnUNet_results/Dataset502_FullDataset_split1_pred",
            "/mnt/Drive2/lucas/models/nnUnet/nnUNet_results/Dataset503_FullDataset_split2_pred",
            "/mnt/Drive2/lucas/models/nnUnet/nnUNet_results/Dataset504_FullDataset_split3_pred"
            ]

    print(f"Collecting nnUnet predictions.")
    pred_files_dict = {}
    for dir_path in result_dirs:
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".nii.gz"):
                key = file_name.replace(".nii.gz", "")
                full_path = os.path.join(dir_path, file_name)
                pred_files_dict[key] = full_path

    print(f"Found {len(pred_files_dict)} nn-Unet predictions")

    all_results = []
    for d_id, d_dir, d_rootdir in zip(DATASET_IDS, DATASET_DIRS, ROOT_DIRS):
        print(f"Evaluating {d_id}...")

        if not d_id in args.dataset.upper() and args.dataset != "all":
            continue

        dataset = LongitudinalDataset(dataset_id=d_id, root_dir=d_rootdir)
        dataset.load(d_dir)

        for patient_ind, patient in enumerate(dataset.patients):
            patient_id = patient["patient_id"]
            p_id_nnunet = dataset.dataset_id + f"_{patient_ind}"
            preop_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]  # Find first preop exam
            followup_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="followup")[0]

            if "UPENN" in d_id:
                preop_exam_dir = preop_exam["t1"].parent
                followup_exam_dir = followup_exam["t1"].parent
            elif "GLIODIL" in d_id:
                preop_exam_dir = preop_exam["t1c"].parent / "preop"
                followup_exam_dir = followup_exam["t1c"].parent / "followup"
            else:
                preop_exam_dir = preop_exam["t1c"].parent
                followup_exam_dir = followup_exam["t1c"].parent

            pred_file = pred_files_dict[p_id_nnunet]
            try:
                results = evaluate_tumor_model(
                        preop_dir=preop_exam_dir,
                        followup_dir=followup_exam_dir,
                        pred_file=pred_file,
                        model_id="nnUnet"
                        )
                all_results.append(results)
                print(f"{p_id_nnunet}: {results}")
            except Exception as e:
                raise e
                #print(f"Exception: {e}")

    recurrence_coverage_standard = [r["recurrence_coverage_standard"] for r in all_results]
    recurrence_coverage_standard_all = [r["recurrence_coverage_standard_all"] for r in all_results]
    recurrence_coverage_model = [r["recurrence_coverage_model"] for r in all_results]
    recurrence_coverage_model_all = [r["recurrence_coverage_model_all"] for r in all_results]

    print(f"Finished evaluation.")
    print(f"Standard plan coverge: {100*np.mean(recurrence_coverage_standard):.2f} \u00B1 {100*stats.sem(recurrence_coverage_standard):.2f}")
    print(f"Standard plan coverge (all): {100*np.mean(recurrence_coverage_standard_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_standard_all):.2f}")
    print(f"Model plan coverge: {100*np.mean(recurrence_coverage_model):.2f} \u00B1 {100*stats.sem(recurrence_coverage_model):.2f}")
    print(f"Model plan coverge (all): {100*np.mean(recurrence_coverage_model_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_model_all):.2f}")
