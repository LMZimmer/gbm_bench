import os
import json
import math
import shutil
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from pathlib import Path
from radiomics.firstorder import RadiomicsFirstOrder
from radiomics.glcm import RadiomicsGLCM
from radiomics.shape import RadiomicsShape
from radiomics.featureextractor import RadiomicsFeatureExtractor
from gbm_bench.utils.utils import merge_pdfs
from gbm_bench.utils.utils import compute_center_of_mass
from gbm_bench.utils.constants import *
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.utils.visualization import plot_tumor_sizes, plot_performances, plot_com_distances
from typing import List, Optional
from pathlib import Path
import matplotlib.pyplot as plt


def get_institute(pat_id):
    if "rhuh" in pat_id.lower():
        return "RHUH"
    if "patient" in pat_id.lower():
        return "LUMIERE"
    if "data" in pat_id.lower():
        return "DATA"
    if "respond" in pat_id.lower():
        return "RESPOND"
    if "tgm" in pat_id.lower():
        return "TGM"
    return "None"


def get_pyradiomics_features(image_dir, mask_dir):
    config_dir = "/home/home/lucas/projects/gbm_bench/pyrad_settings.yaml"
    extractor = RadiomicsFeatureExtractor(config_dir)
    breakpoint()
    pyrad_feats = extractor.execute(image_dir, mask_dir, label=3) #NOTE: returns dict name: feature
    return pyrad_feats


def get_pyradiomics_features_direct(image_dir, mask_dir):
    #NOTE: mask should be tumor mask
    #image = np.rint(nib.load(image_dir).get_fdata()).astype(np.int32)
    #mask = np.rint(nib.load(mask_dir).get_fdata()).astype(np.int32)
    #mask = (mask > 0).astype(np.int32)
    image = nib.load(image_dir)
    mask = nib.load(mask_dir)
    first = RadiomicsFirstOrder(image, mask)
    glcm = RadiomicsGLCM(image, mask)
    shape = RadiomicsShape(image, mask)
    feature_names = ["Energy", "Total Energy", "Entropy", "Min", "10percentile", "90percentile", "max", "median", "mean_abs_dev", "mean_abs_dev_robust", "RMS", "StdDev", "skewness", "kurtosis", "variance", "uniformity", "Autocorrelation", "JointAverage", "ClusterProminence", "ClusterShade", "ClusterTendency", "Contrast", "Correlation", "DiffAverage", "DiffEntropy", "DiffVariance", "JointEnergy", "JointEntropy", "IMC1", "IMC2", "IDM", "IDMN", "IdFeature", "IdnFeature", "InverseVariance", "MaxProbability", "SumEntropy", "SumSquared", "Sphericity", "MaxDiameter"]
    features = [
            first.getEnergyFeatureValue(),
            first.getTotalEnergyFeatureValue(),
            first.getEntropyFeatureValue(),
            first.getMinimumFeatureValue(),
            first.get10PercentileFeatureValue(),
            first.get90PercentileFeatureValue(),
            first.getMaximumFeatureValue(),
            first.getMeanFeatureValue(),
            first.getMedianFeatureValue(),
            first.getMeanAbsoluteDeviationFeatureValue(),
            first.getRobustMeanAbsoluteDeviationFeatureValue(),
            first.getRootMeanSquaredFeatureValue(),
            first.getStandardDeviationFeatureValue(),
            first.getSkewnessFeatureValue(),
            first.getKurtosisFeatureValue(),
            first.getVarianceFeatureValue(),
            first.getUniformityFeatureValue(),
            glcm.getAutocorrelationFeatureValue(),
            glcm.getJointAverageFeatureValue(),
            glcm.getClusterProminenceFeatureValue(),
            glcm.getClusterShadeFeatureValue(),
            glcm.getClusterTendencyFeatureValue(),
            glcm.getContrastFeatureValue(),
            glcm.getCorrelationFeatureValue(),
            glcm.getDifferenceAverageFeatureValue(),
            glcm.getDifferenceEntropyFeatureValue(),
            glcm.getDifferenceVarianceFeatureValue(),
            glcm.getJointEnergyFeatureValue(),
            glcm.getJointEntropyFeatureValue(),
            glcm.getImc1FeatureValue(),
            glcm.getImc2FeatureValue(),
            glcm.getIdmFeatureValue(),
            glcm.getIdmnFeatureValue(),
            glcm.getIdFeatureValue(),
            glcm.getIdnFeatureValue(),
            glcm.getInverseVarianceFeatureValue(),
            glcm.getMaximumProbabilityFeatureValue(),
            glcm.getSumEntropyFeatureValue(),
            glcm.getSumSquaresFeatureValue(),
            shape.getSphericityFeatureValue(),
            shape.getMaximum3DDiameterFeatureValue()
            ]
    return features, feature_names


def get_tumorsize(tumorpath):
    tumorseg = np.rint(nib.load(tumorseg_dir).get_fdata()).astype(np.int32)
    necrosis_size = np.sum((tumorseg==1).astype(np.int32)) / 1000
    edema_size = np.sum((tumorseg==2).astype(np.int32)) / 1000
    enhancing_size = np.sum((tumorseg==3).astype(np.int32)) / 1000
    return (necrosis_size, edema_size, enhancing_size)


def normalize(lst):
    if not lst:
        return []  # handle empty list

    min_val = min(lst)
    max_val = max(lst)

    # Avoid division by zero if all values are the same
    if max_val == min_val:
        return [0.0 for _ in lst]

    return [(x - min_val) / (max_val - min_val) for x in lst]


def get_com(tumorpath):
    tumorseg = np.rint(nib.load(tumorseg_dir).get_fdata()).astype(np.int32)
    com_core = compute_center_of_mass(tumorseg, tumorseg, classes=[1,3])
    com_full = compute_center_of_mass(tumorseg, tumorseg, classes=[1,2,3])
    return (com_core, com_full)


def get_resolution(nifti_path: Path) -> float:
    """
    Compute the mean voxel resolution (in mm) of a NIfTI image
    directly from its affine transformation matrix.

    Parameters
    ----------
    nifti_path : Path
        Path to the NIfTI file (.nii or .nii.gz).

    Returns
    -------
    float
        Mean voxel size (in millimeters).
    """
    # Load the image
    img = nib.load(str(nifti_path))
    affine = img.affine

    # Extract voxel sizes from the affine matrix (absolute norm of each column vector)
    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    # Return the mean voxel size
    return float(np.mean(voxel_sizes))


def intensity_entropy(nifti_path: Path, num_bins: int = 256) -> float:
    """
    Compute the Shannon intensity entropy of a NIfTI image.

    Parameters
    ----------
    nifti_path : Path
        Path to the NIfTI file (.nii or .nii.gz).
    num_bins : int, optional
        Number of histogram bins to use (default=256).

    Returns
    -------
    float
        Shannon entropy of the image intensity distribution (in bits).
    """
    # Load NIfTI file
    img = nib.load(str(nifti_path))
    data = img.get_fdata()

    # Flatten and remove NaNs or zeros if desired
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.nan

    # Compute normalized histogram of intensities
    hist, bin_edges = np.histogram(data, bins=num_bins, density=True)
    hist = hist[hist > 0]  # remove zero bins to avoid log(0)

    # Shannon entropy (base 2 = bits)
    H = stats.entropy(hist, base=2)
    return float(H)


def plot_histograms(
    data: List[List[float]],
    save_path: Path,
    labels: Optional[List[str]] = None
) -> None:
    """
    Plots multiple histograms (one for each list of floats) on the same figure,
    each with an opacity of 0.5, and saves the figure to the given path.

    Args:
        data (List[List[float]]): A list of lists, where each inner list contains float values.
        save_path (Path): The path (including filename) where the plot will be saved.
        labels (Optional[List[str]]): A list of labels for each dataset.
            If None or shorter than 'data', generic labels are used.
    """
    if not data:
        raise ValueError("Input 'data' must not be empty.")

    plt.figure(figsize=(8, 6))

    for i, values in enumerate(data):
        if not values:
            continue  # skip empty datasets

        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"
        plt.hist(values, bins=100, alpha=0.5, label=label)

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Overlapping Histograms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # Example:
    # nohup python -u scripts/identify_subsets.py -algorithm gliodil 1>csv_output.txt 2>&1 &
    # nohup python -u scripts/identify_subsets.py -algorithm sbtc 1>csv_output.txt 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-algorithm", type=str, help="Algorithm ID to evaluate.")
    args = parser.parse_args()

    save_root = Path("/home/home/lucas/projects/gbm_bench/tmp_visualization/")

    DATASET_IDS = [
            "RHUH",
            "LUMIERE",
            "GLIODIL",
            ]
    DATASET_DIRS = [
            RHUH_GBM_DIR,
            LUMIERE_DIR,
            GLIODIL_DIR,
            ]
    ROOT_DIRS = [
            "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM",
            "/mnt/Drive2/lucas/datasets/LUMIERE/Imaging",
            "/mnt/Drive2/lucas/datasets/GLIODIL",
            ]

    print(f"Evaluating {args.algorithm}")
    all_results = []
    core_coms = []
    full_coms = []
    nec_sizes = []
    edema_sizes = []
    enh_sizes = []
    lmi_coefficients = []
    gliodil_coefficients = []
    patient_ids = []
    entropy_t1c = []
    entropy_flair = []
    resolution_t1c = []
    resolution_flair = []
    institute = []
    pyrad_features = []

    for d_id, d_dir, d_rootdir in zip(DATASET_IDS, DATASET_DIRS, ROOT_DIRS):
        print(f"Evaluating {d_id}...")

        dataset = LongitudinalDataset(dataset_id=d_id, root_dir=d_rootdir)
        dataset.load(d_dir)

        for patient_ind, patient in enumerate(dataset.patients[0:1]): #TODO
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
                if "rhuh" in patient_id.lower():
                    t1c_dir = MODALITY_CONVERTED_SCHEMA.format(base_dir=preop_exam_dir, modality="t1c")
                    flair_dir = MODALITY_CONVERTED_SCHEMA.format(base_dir=preop_exam_dir, modality="flair")
                else:
                    preop_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]
                    t1c_dir = preop_exam["t1c"]
                    flair_dir = preop_exam["flair"]
                performance_dir = METRICS_SCHEMA.format(base_dir=followup_exam_dir, algo_id=args.algorithm)
                tumorseg_dir = TUMORSEG_SCHEMA.format(base_dir=preop_exam_dir)
                gliodil_coeffs_dir = PREDICTION_OUTPUT_SCHEMA.format(base_dir=preop_exam_dir, algo_id="gliodil").parent / "coeffs.npy"
                lmi_coeffs_dir = PREDICTION_OUTPUT_SCHEMA.format(base_dir=preop_exam_dir, algo_id="lmi").parent / "lmi_parameters.npy"
                
                performance_dict = json.load(open(performance_dir, "r"))
                tumor_size = get_tumorsize(tumorseg_dir)
                com = get_com(performance_dir)
                ent_t1c = intensity_entropy(t1c_dir)
                ent_flair = intensity_entropy(flair_dir)
                res_t1c = get_resolution(t1c_dir)
                res_flair = get_resolution(flair_dir)
                inst = get_institute(patient_id)
                #pyrad_feats, pyrad_feat_names = get_pyradiomics_features(t1c_dir, tumorseg_dir)
                pyrad_feats = get_pyradiomics_features(t1c_dir, tumorseg_dir)

                try:
                    gliodil_coeffs = np.load(gliodil_coeffs_dir).tolist()
                    if any(math.isnan(x) for x in gliodil_coeffs):
                        gliodil_coeffs[2] = float(com[0][0])
                        gliodil_coeffs[3] = float(com[0][1])
                        gliodil_coeffs[4] = float(com[0][2])
                    print(gliodil_coeffs)
                    gliodil_coefficients.append(gliodil_coeffs)
                except:
                    gliodil_coefficients.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

                try:
                    lmi_coeffs = np.load(lmi_coeffs_dir, allow_pickle=True).item()
                    lmi_coefficients.append(lmi_coeffs)
                except:
                    lmi_coefficients.append({"D": 0., "rho": 0., "T": 0., "x": 0., "y": 0., "z": 0.})
                
                patient_ids.append(patient_id)
                all_results.append(performance_dict)
                nec_sizes.append(tumor_size[0])
                edema_sizes.append(tumor_size[1])
                enh_sizes.append(tumor_size[2])
                core_coms.append(com[0])
                full_coms.append(com[1])
                entropy_t1c.append(ent_t1c)
                entropy_flair.append(ent_flair)
                resolution_t1c.append(res_t1c)
                resolution_flair.append(res_flair)
                institute.append(inst)
                pyrad_features.append(pyrad_feats)
            except Exception as e:
                print(f"Exception for {patient_id}: {e}")
        
        #cov_mod = [r["recurrence_coverage_model"] for r in dataset_results]

    recurrence_coverage_standard = [r["recurrence_coverage_standard"] for r in all_results]
    recurrence_coverage_standard_all = [r["recurrence_coverage_standard_all"] for r in all_results]
    recurrence_coverage_model = [r["recurrence_coverage_model"] for r in all_results]
    recurrence_coverage_model_all = [r["recurrence_coverage_model_all"] for r in all_results]

    improv_core = [r["recurrence_coverage_model"] - r["recurrence_coverage_standard"] for r in all_results]
    improv_full = [r["recurrence_coverage_model_all"] - r["recurrence_coverage_standard_all"] for r in all_results]

    lmi_D = normalize([coefs["D"] for coefs in lmi_coefficients])
    lmi_rho = normalize([coefs["rho"] for coefs in lmi_coefficients])
    lmi_T = normalize([coefs["T"] for coefs in lmi_coefficients])
    lmi_x = normalize([coefs["x"] for coefs in lmi_coefficients])
    lmi_y = normalize([coefs["y"] for coefs in lmi_coefficients])
    lmi_z = normalize([coefs["z"] for coefs in lmi_coefficients])

    gliodil_D = normalize([coefs[0] for coefs in gliodil_coefficients])
    gliodil_f = normalize([coefs[1] for coefs in gliodil_coefficients])
    gliodil_x = normalize([coefs[2] for coefs in gliodil_coefficients])
    gliodil_y = normalize([coefs[3] for coefs in gliodil_coefficients])
    gliodil_z = normalize([coefs[4] for coefs in gliodil_coefficients])

    com_x = normalize([cc[0] for cc in core_coms])
    com_y = normalize([cc[1] for cc in core_coms])
    com_z = normalize([cc[2] for cc in core_coms])

    com_all_x = normalize([cc[0] for cc in full_coms])
    com_all_y = normalize([cc[1] for cc in full_coms])
    com_all_z = normalize([cc[2] for cc in full_coms])

    entropy_t1c = normalize(entropy_t1c)
    entropy_flair = normalize(entropy_flair)

    #pyrad_feats_reshaped = [[feats[ind] for feats in pyrad_features] for ind in range(len(pyrad_feat_names))]
    pyrad_feat_names = [k for k in pyrad_features[0].keys()]
    pyrad_feats_reshaped = [[pf[ind][k] for ind in range(len(pyrad_features))] for k in pyrad_feat_names]

    colnames = ["PatientID", "source", "res t1c", "res flair", "entropy t1c", "entropy flair", "nec size [cm3]", "edema size [cm3]", "enh size [cm3]", "com (core) x", "com (core) y", "com (core) z", "com (all) x", "com (all) y", "com (all) z", "D (lmi)", "rho(lmi)", "T (lmi)", "x (lmi)", "y (lmi)", "z (lmi)", "D", "f", "x0", "y0", "z0", "cov (std)", f"cov (args.algorithm)", "cov full (std)", f"cov full (args.algorithm)", "improvement core", "improvement all"] + pyrad_feat_names
    
    data = [patient_ids, institute, resolution_t1c, resolution_flair, entropy_t1c, entropy_flair, nec_sizes, edema_sizes, enh_sizes, com_x, com_y, com_z, com_all_x, com_all_y, com_all_z, lmi_D, lmi_rho, lmi_T, lmi_x, lmi_y, lmi_z, gliodil_D, gliodil_f, gliodil_x, gliodil_y, gliodil_z, recurrence_coverage_standard, recurrence_coverage_model, recurrence_coverage_standard_all, recurrence_coverage_model_all, improv_core, improv_full] + pyrad_feats_reshaped
    df = pd.DataFrame(data).T
    df.columns = colnames
    df.to_csv("predict_gbm_meta.csv", index=False)

    plot_histograms(
            data=[recurrence_coverage_standard, recurrence_coverage_model],
            save_path=save_root / "coverage_hist.pdf",
            labels=["standard", f"{args.algorithm}"]
            )
    plot_histograms(
            data=[recurrence_coverage_standard_all, recurrence_coverage_model_all],
            save_path=save_root / "coverage_all_hist.pdf",
            labels=["standard", f"{args.algorithm}"]
            )
    plot_histograms(
            data=[improv_core, improv_full],
            save_path=save_root / "improvement_hist.pdf",
            labels=["core", f"full"]
            )
