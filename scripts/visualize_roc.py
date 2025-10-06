import os
import json
import shutil
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from scipy import stats
from pathlib import Path
from gbm_bench.utils.constants import *
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.utils.visualization import plot_tumor_sizes, plot_performances, plot_com_distances


def save_mean_roc_with_band(fprs, tprs, out_path="mean_roc.png", n_grid=1000):
    """
    Compute a mean ROC from multiple ROC curves and save a figure that plots:
      • the mean ROC curve, and
      • a shaded ±1 SD band around it (no individual curves).

    Parameters
    ----------
    fprs : List[List[float]]
    tprs : List[List[float]]
    out_path : str
    n_grid : int

    Returns
    -------
    dict with JSON-friendly contents:
      {
        "mean_fpr": List[float],
        "mean_tpr": List[float],
        "tpr_sd":   List[float],
        "auc_mean": float,         # mean of per-curve AUCs (over valid curves)
        "auc_sd":   float,         # SD of per-curve AUCs
        "auc_mean_curve": float,   # AUC of the mean ROC curve
        "n_curves_in": int,
        "n_curves_used": int,
        "n_curves_skipped": int
      }
    """
    if len(fprs) != len(tprs):
        raise ValueError("fprs and tprs must have the same number of curves.")
    if len(fprs) == 0:
        raise ValueError("No curves provided.")

    mean_fpr = np.linspace(0.0, 1.0, int(n_grid))
    interp_tprs = []
    aucs = []
    n_skipped = 0

    for fpr_i, tpr_i in zip(fprs, tprs):
        fpr = np.asarray(fpr_i, dtype=float).ravel()
        tpr = np.asarray(tpr_i, dtype=float).ravel()

        if fpr.size != tpr.size or fpr.size < 2:
            n_skipped += 1
            continue

        # drop non-finite
        finite_mask = np.isfinite(fpr) & np.isfinite(tpr)
        fpr = fpr[finite_mask]
        tpr = tpr[finite_mask]
        if fpr.size < 2:
            n_skipped += 1
            continue

        # clip to [0,1]
        fpr = np.clip(fpr, 0.0, 1.0)
        tpr = np.clip(tpr, 0.0, 1.0)

        # sort by FPR and drop duplicates
        order = np.argsort(fpr)
        fpr = fpr[order]; tpr = tpr[order]
        fpr_unique, idx = np.unique(fpr, return_index=True)
        tpr_unique = tpr[idx]

        if fpr_unique.size < 2:
            n_skipped += 1
            continue

        # enforce endpoints for stable interpolation
        if fpr_unique[0] > 0.0:
            fpr_unique = np.insert(fpr_unique, 0, 0.0)
            tpr_unique = np.insert(tpr_unique, 0, 0.0)
        if fpr_unique[-1] < 1.0:
            fpr_unique = np.append(fpr_unique, 1.0)
            tpr_unique = np.append(tpr_unique, 1.0)

        # interpolate onto common grid
        interp = np.interp(mean_fpr, fpr_unique, tpr_unique)
        interp[0] = 0.0
        interp[-1] = 1.0
        if not np.all(np.isfinite(interp)):
            n_skipped += 1
            continue

        interp_tprs.append(interp)
        aucs.append(auc(fpr_unique, tpr_unique))

    if len(interp_tprs) == 0:
        raise ValueError(
            "All input ROC curves were invalid after cleaning. "
            "Check for NaNs/Infs, unsorted FPR, or degenerate curves."
        )

    interp_tprs = np.vstack(interp_tprs)
    mean_tpr = interp_tprs.mean(axis=0)
    std_tpr  = interp_tprs.std(axis=0)
    auc_mean = float(np.mean(aucs))
    auc_sd   = float(np.std(aucs))
    auc_mean_curve = float(auc(mean_fpr, mean_tpr))

    # Plot mean curve + ±1 SD band (no individual curves)
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, linewidth=2,
             label=f"Mean ROC (AUC={auc_mean:.3f}±{auc_sd:.3f})")
    tpr_lower = np.clip(mean_tpr - std_tpr, 0.0, 1.0)
    tpr_upper = np.clip(mean_tpr + std_tpr, 0.0, 1.0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.2, label="±1 SD")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC ± 1 SD")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return {
        "mean_fpr": mean_fpr.tolist(),
        "mean_tpr": mean_tpr.tolist(),
        "tpr_sd": std_tpr.tolist(),
        "auc_mean": auc_mean,
        "auc_sd": auc_sd,
        "auc_mean_curve": auc_mean_curve,
        "n_curves_in": len(fprs),
        "n_curves_used": int(interp_tprs.shape[0]),
        "n_curves_skipped": int(n_skipped),
    }


if __name__ == "__main__":
    # Example:
    # python scripts/visualize_roc.py -algorithm sbtc
    parser = argparse.ArgumentParser()
    parser.add_argument("-algorithm", type=str, help="Algorithm ID to evaluate.")
    args = parser.parse_args()

    outfile = f"tmp_visualization/roc_curve_{args.algorithm}.pdf"
    outfile_standard = f"tmp_visualization/roc_curve_standard.pdf"
    outfile_standard_fade = f"tmp_visualization/roc_curve_standard_fade.pdf"

    DATASET_IDS = [
            "RHUH",
            #"UPENN",
            "LUMIERE",
            "GLIODIL",
            #"IVYGAP",
            #"CPTAC",
            #"TCGA-GBM"
            ] #"TCGA-LGG"]
    DATASET_DIRS = [
            RHUH_GBM_DIR,
            #UPENN_GBM_DIR,
            LUMIERE_DIR,
            GLIODIL_DIR,
            #IVYGAP_DIR,
            #CPTAC_DIR,
            #TCGA_GBM_DIR
            ] #TCGA_LGG_DIR]
    ROOT_DIRS = [
            "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM",
            #"/home/home/lucas/data/UPENN-GBM/UPENN-GBM",
            "/mnt/Drive2/lucas/datasets/LUMIERE/Imaging",
            "/mnt/Drive2/lucas/datasets/GLIODIL",
            #"/mnt/Drive2/lucas/datasets/IVYGAP",
            #"/mnt/Drive2/lucas/datasets/CPTAC-GBM",
            #"/mnt/Drive2/lucas/datasets/TCGA-GBM",
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
                
                if "tpr_model" in performance_dict.keys() and "fpr_standard_fade" in performance_dict.keys():
                    all_results.append(performance_dict)
                    dataset_results.append(performance_dict)
                else:
                    print(performance_dict.keys())
            except Exception as e:
                print(f"Exception for {patient_id}: {e}")

    import math
    sensitivity_model = [r["sensitivity_model"] for r in all_results]
    specificity_model = [r["specificity_model"] for r in all_results if not math.isnan(r["specificity_model"])]
    sensitivity_standard = [r["sensitivity_standard"] for r in all_results]
    specificity_standard = [r["specificity_standard"] for r in all_results if not math.isnan(r["specificity_standard"])]

    print(f"Mean sensitivity (model): {np.mean(sensitivity_model):.3f} \u00B1 {stats.sem(sensitivity_model):.3f}")
    print(f"Mean specificity (model): {np.mean(specificity_model):.3f} \u00B1 {stats.sem(specificity_model):.3f}")
    print(f"Mean sensitivity (standard): {np.mean(sensitivity_standard):.3f} \u00B1 {stats.sem(sensitivity_standard):.3f}")
    print(f"Mean specificity (standard): {np.mean(specificity_standard):.3f} \u00B1 {stats.sem(specificity_standard):.3f}")

    fprs_model = [r["fpr_model"] for r in all_results]
    tprs_model = [r["tpr_model"] for r in all_results]

    fprs_standard = [r["fpr_standard"] for r in all_results]
    tprs_standard = [r["tpr_standard"] for r in all_results]

    fprs_standard_fade = [r["fpr_standard_fade"] for r in all_results]
    tprs_standard_fade = [r["tpr_standard_fade"] for r in all_results]

    # All curves
    """
    plt.figure()
    for fpr, tpr in zip(fprs_model, tprs_model):
        idx = np.linspace(0, len(fpr) - 1, 1000, dtype=int)
        plt.plot(np.array(fpr)[idx], np.array(tpr)[idx])
    
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    """

    # Mean curve
    save_mean_roc_with_band(fprs_standard, tprs_standard, out_path=outfile_standard, n_grid=1000)
    save_mean_roc_with_band(fprs_standard_fade, tprs_standard_fade, out_path=outfile_standard_fade, n_grid=1000)
    output = save_mean_roc_with_band(fprs_model, tprs_model, out_path=outfile, n_grid=1000)
    print(f"Curves used: {output['n_curves_used']}")
    print(f"Curves discarded: {output['n_curves_skipped']}")
