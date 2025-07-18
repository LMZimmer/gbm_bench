import os
import json
import shutil
import argparse
import numpy as np
import nibabel as nib
from scipy import stats
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from gbm_bench.utils.constants import *
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.utils.utils import load_mri_data, load_and_resample_mri_data



def roc_auc(pred_file, seg_file, mask=None, labels_of_interest=[1, 3], drop_intermediate=True):

    # Load
    seg = np.rint(load_mri_data(str(seg_file))).astype(np.int32)
    pred = load_and_resample_mri_data(str(pred_file), resample_params=seg.shape, interp_type=0)

    if mask is None:
        mask = np.ones_like(seg, dtype=bool)

    # Flatten only voxels under mask
    scores = pred[mask].ravel()
    labels = seg[mask].ravel()

    # Get recurrence core according to labels of interest
    y_true = np.isin(labels, labels_of_interest).astype(int)

    # Guard against degenerate cases
    pos = y_true.sum()
    neg = y_true.size - pos
    if pos == 0 or neg == 0:
        raise ValueError("The masked region must contain both positive and negative voxels.")

    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=drop_intermediate)
    auc_value = auc(fpr, tpr)

    return auc_value, fpr, tpr, thresholds



if __name__ == "__main__":
    # Example:
    # python scripts/roc_auc.py -algorithm sbtc > sbtc_auc.txt
    # python scripts/roc_auc.py -algorithm gliodil > gliodil_auc.txt
    # python scripts/roc_auc.py -algorithm lmi > lmi_auc.txt
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
                seg_file = RECURRENCE_SCHEMA.format(base_dir=str(followup_exam_dir))
                pred_file = PREDICTION_OUTPUT_SCHEMA.format(base_dir=str(preop_exam_dir), algo_id=args.algorithm)

                auc_value, fpr, tpr, thresholds = roc_auc(pred_file=pred_file, seg_file=seg_file)

                dataset_results.append(auc_value)
                all_results.append(auc_value)
            except Exception as e:
                print(f"Exception for {patient_id}: {e}")
        print(f"{d_id}: {np.mean(dataset_results):.2f} \u00B1 {stats.sem(dataset_results):.2f}")
    print(f"{np.mean(all_results):.2f} \u00B1 {stats.sem(all_results):.2f}")
