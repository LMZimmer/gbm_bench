import os
import json
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from loguru import logger
from typing import Any, Dict
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import center_of_mass, distance_transform_edt
from gbm_bench.evaluation.metrics import coverage
from gbm_bench.utils.utils import load_mri_data, load_and_resample_mri_data, is_binary_array
from gbm_bench.utils.constants import (
        BRAIN_MASK_SCHEMA,
        METRICS_SCHEMA,
        MODALITY_STRIPPED_SCHEMA,
        MODEL_PLAN_SCHEMA,
        RECURRENCE_SCHEMA,
        STANDARD_PLAN_SCHEMA,
        TISSUE_SEG_SCHEMA,
        TISSUE_LABELS,
        TUMORSEG_SCHEMA,
        TUMORSEG_CORE_SCHEMA
        )


def sensitivity(tumor_seg: np.ndarray, radiation_plan: np.ndarray) -> float:
    """
    Compute sensitivity = TP / (TP + FN)
    where:
    - TP = number of voxels correctly predicted as tumor (1 in both)
    - FN = number of voxels in tumor_seg that were missed (1 in tumor_seg, 0 in radiation_plan)
    Returns float in [0,1]. If there are no positive voxels in tumor_seg, returns np.nan or 0 (your choice).
    """
    # ensure boolean
    gt = (tumor_seg > 0)
    pred = (radiation_plan > 0)

    TP = np.logical_and(gt, pred).sum()
    FN = np.logical_and(gt, np.logical_not(pred)).sum()

    denom = TP + FN
    if denom == 0:
        # no positives in ground truth; sensitivity is undefined or we return 0 / nan
        return float('nan')
    return TP / denom


def specificity(tumor_seg: np.ndarray, radiation_plan: np.ndarray) -> float:
    """
    Compute specificity = TN / (TN + FP)
    where:
    - TN = number of voxels correctly predicted as background (0 in both)
    - FP = number of voxels incorrectly predicted as tumor (0 in tumor_seg, 1 in radiation_plan)
    Returns float in [0,1]. If there are no negative voxels in tumor_seg, returns np.nan or 0 (your choice).
    """
    gt = (tumor_seg > 0)
    pred = (radiation_plan > 0)

    TN = np.logical_and(~gt, ~pred).sum()
    FP = np.logical_and(~gt, pred).sum()

    denom = TN + FP
    if denom == 0:
        # no negatives in ground truth; specificity undefined
        return float('nan')
    return TN / denom


def create_standard_plan(core_segmentation: np.ndarray, ctv_margin: int) -> np.ndarray:
    """
    Creates a target volume mask by dilating the tumor core segmentation with ctv_margin
    
    Parameters:
        core_segmentation (np.ndarray): A NumPy array representing the core segmentation mask, where non-zero
            values indicate the region of interest.
        ctv_margin (int): The margin to dilate the core segmentation.

    Returns:
        np.ndarray: A binary NumPy array of the same shape as core_segmentation, where
            pixels within the ctv_margin from the core segmentation are True, and
            all other pixels are False.
    """
    if ctv_margin <= 0:
        raise ValueError("ctv_margin must be a positive int.")
    distance_transform = distance_transform_edt(~ (core_segmentation >0))
    dilated_core = distance_transform <= ctv_margin
    return dilated_core.astype(np.int32)


def find_threshold(volume: np.ndarray, target_volume: float, tolerance: float = 0.01, initial_threshold: float = 0.2, maxIter: int = 10000):
    """
    Compute the threshold that produces a specified volume after masking the input array using the threshold.

    Parameters:
        volume (np.ndarray): A NumPy array representing the input volume to be thresholded (tumor cell concentration).
        target_volume (float): The desired volume size.
        tolerance (float, optional): The acceptable relative difference between the target volume and the thresholded volume. Default is 0.01 (1%).
        initial_threshold (float, optional): The starting threshold value for the segmentation. Should be between 0 and 1.
        max_iter (int, optional): The maximum number of iterations to perform. If the threshold is not found within this number of iterations,
            the function will terminate.

    Returns:
        float: The threshold value that segments the volume to match the target volume within the specified tolerance.
            Returns 1.01 if the threshold exceeds the valid range (0, 1), indicating an "above the model" condition.
            Returns 0 if the maximum number of iterations is reached without finding a suitable threshold.
    """

    if np.sum(volume > 0) < target_volume:
        print("Volume is too small")
        return 0

    # Define the initial threshold, step, and previous direction
    threshold = initial_threshold
    step = 0.1
    previous_direction = None

    # Calculate the current volume
    current_volume = np.sum(volume > threshold)

    # Iterate until the current volume is within the tolerance of the target volume

    while abs(current_volume - target_volume) / target_volume > tolerance:
        # Determine the current direction
        if current_volume > target_volume:
            direction = 'increase'
        else:
            direction = 'decrease'

        # Adjust the threshold
        if direction == 'increase':
            threshold += step
        else:
            threshold -= step

        # Check if the threshold is out of bounds
        if threshold < 0 or threshold > 1:
            return 1.01 #above the model

        # Update the current volume
        current_volume = np.sum(volume > threshold)

        # Reduce the step size if the direction has alternated
        if previous_direction and previous_direction != direction:
            step *= 0.5

        # Update the previous direction
        previous_direction = direction

        maxIter -= 1
        if maxIter < 0:
            print("Max Iter reached, no threshold found")
            return 0

    return threshold


def recurrence_coverage(recurrence_segmentation: np.ndarray, target_volume: np.ndarray) -> float:
    """
    Calculate the coverage of tumor recurrence by the treatment plan volume.

    Parameters:
        recurrence_segmentation (np.ndarray): A (boolean) NumPy array indicating the presence of tumor recurrence.
        target_volume (np.ndarray): A (boolean) NumPy array indicating the area covered by the treatment plan.

    Returns:
        float: The coverage ratio of the treatment plan over the tumor recurrence (0-1.0).
            Returns 1.0 if there is no tumor recurrence.
    """
    if not is_binary_array(recurrence_segmentation):
        raise ValueError(f"recurrence_segmentation values have to be in (True, False, 0, 1, 0.0, 1.0).")
    if not is_binary_array(target_volume):
        raise ValueError(f"target_volume values have to be in (True, False, 0, 1, 0.0, 1.0).")
    if recurrence_segmentation.shape != target_volume.shape:
        raise ValueError(f"Dimension mismatch between recurrence_segmentation and target_volume.")
    
    # If there is no recurrence, return 1
    if np.sum(recurrence_segmentation) <=  0.00001:
        return 1

    # Calculate the intersection between the recurrence and the plan
    intersection = np.logical_and(recurrence_segmentation, target_volume)

    # Calculate the coverage as the ratio of the intersection to the recurrence
    coverage = np.sum(intersection) / np.sum(recurrence_segmentation)
    return coverage


def missed_voxels(recurrence_segmentation: np.ndarray, target_volume: np.ndarray) -> int:
    """Count recurrence voxels not covered by a treatment plan.

    Parameters
    ----------
    recurrence_segmentation : np.ndarray
        Boolean array marking the observed recurrence region.
    target_volume : np.ndarray
        Boolean array representing the treatment plan volume.

    Returns
    -------
    int
        Number of recurrence voxels outside of the treatment plan volume.
    """
    if not is_binary_array(recurrence_segmentation):
        raise ValueError(f"recurrence_segmentation values have to be in (True, False, 0, 1, 0.0, 1.0).")
    if not is_binary_array(target_volume):
        raise ValueError(f"target_volume values have to be in (True, False, 0, 1, 0.0, 1.0).")
    if recurrence_segmentation.shape != target_volume.shape:
        raise ValueError(f"Dimension mismatch between recurrence_segmentation and target_volume.")

    missed = np.logical_and(recurrence_segmentation, np.logical_not(target_volume))
    return int(np.sum(missed))


def generate_distance_fade_mask(binary_model_prediction: np.ndarray) -> np.ndarray:
    if not is_binary_array(binary_model_prediction):
        raise ValueError("Model prediction is not binary: {np.unique(binary_model_prediction)}")

    data = np.rint(binary_model_prediction).astype(np.int32)

    # Compute distance transform on background
    distance = distance_transform_edt(data == 0)

    # Normalize distances to [0, 1] and invert: closer to mask = higher value
    max_dist = np.max(distance) if distance.max() > 0 else 1.0
    fade = 1 - (distance / max_dist)

    fade[data == 1] = 1  # Ensure mask stays at 1
    return fade.astype(np.float32)


def generate_distance_fade_mask_no_plateau(binary_model_prediction: np.ndarray, visible_tumor_threshold: float = 0.6) -> np.ndarray:
    if not is_binary_array(binary_model_prediction):
        raise ValueError("Model prediction is not binary: {np.unique(binary_model_prediction)}")

    data = np.rint(binary_model_prediction).astype(np.int32)

    distance_outer = distance_transform_edt(data == 0)
    distance_inner = distance_transform_edt(data != 0)

    max_outer = np.max(distance_outer)
    max_inner = np.max(distance_inner)

    distance_outer = distance_outer / max_outer
    distance_inner = distance_inner / max_inner

    #fade = 1 - distance_outer
    #fade[data == 1] = 1 + distance_inner[data==1]
    #fade = fade / 2.
    fade = (1 - distance_outer) * visible_tumor_threshold  # fade from threshold to 0 outside
    fade[data == 1] = visible_tumor_threshold + distance_inner[data==1] * (1 - visible_tumor_threshold)  # fade from 1 to threshold inside
    return fade.astype(np.float32)    


def roc_auc(pred, seg, mask=None, labels_of_interest=[1, 3], drop_intermediate=True, threshold_range=(0.20, 0.70)):
    
    if mask is None:
        mask = np.ones_like(seg, dtype=bool)
    mask_bool = mask.astype(bool)

    # Flatten only voxels under mask
    scores = pred[mask_bool].ravel()
    labels = seg[mask_bool].ravel()

    # Get recurrence core according to labels of interest
    y_true = np.isin(labels, labels_of_interest).astype(int)

    # Guard against degenerate cases
    pos = y_true.sum()
    neg = y_true.size - pos
    if pos == 0 or neg == 0:
        return 0.0, None, None, None
        #raise ValueError("The masked region must contain both positive and negative voxels.")

    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=drop_intermediate)
    auc_value = auc(fpr, tpr)

    # restricted
    """
    thr_min, thr_max = threshold_range
    sel = (thresholds >= thr_min) & (thresholds <= thr_max)
    fpr_sel, tpr_sel, thr_sel = fpr[sel], tpr[sel], thresholds[sel]

    thr_asc, fpr_asc, tpr_asc = thresholds[::-1], fpr[::-1], tpr[::-1]

    if not np.isclose(thr_sel[0] if thr_sel.size else np.inf, thr_min):
        fpr_low = np.interp(thr_min, thr_asc, fpr_asc)
        tpr_low = np.interp(thr_min, thr_asc, tpr_asc)
        fpr_sel = np.insert(fpr_sel, 0, fpr_low)
        tpr_sel = np.insert(tpr_sel, 0, tpr_low)
        thr_sel = np.insert(thr_sel, 0, thr_min)

    if not np.isclose(thr_sel[-1] if thr_sel.size else -np.inf, thr_max):
        fpr_high = np.interp(thr_max, thr_asc, fpr_asc)
        tpr_high = np.interp(thr_max, thr_asc, tpr_asc)
        fpr_sel = np.append(fpr_sel, fpr_high)
        tpr_sel = np.append(tpr_sel, tpr_high)
        thr_sel = np.append(thr_sel, thr_max)

    auc_value = auc(fpr_sel, tpr_sel)
    return auc_value, fpr_sel, tpr_sel, thr_sel
    """

    return auc_value, fpr, tpr, thresholds


def partial_roc_auc(pred, seg, mask=None, labels_of_interest=[1, 3], drop_intermediate=True, threshold_range=(0.0, 0.7), normalize=True):
    if mask is None:
        mask = np.ones_like(seg, dtype=bool)
    mask_bool = mask.astype(bool)

    # Flatten only voxels under mask
    scores = pred[mask_bool].ravel()
    labels = seg[mask_bool].ravel()

    # Get binary labels for classes of interest
    y_true = np.isin(labels, labels_of_interest).astype(int)

    # Guard against degenerate cases
    pos = y_true.sum()
    neg = y_true.size - pos
    if pos == 0 or neg == 0:
        return 0.0, None, None, None

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=drop_intermediate)

    # Filter by threshold range (note: thresholds are sorted descending)
    t_min, t_max = threshold_range
    valid_mask = (thresholds <= t_max) & (thresholds >= t_min)

    if valid_mask.sum() < 2:
        return 0.0, None, None, None

    # Sort FPR/TPR in increasing FPR order for correct integration
    fpr_sel = fpr[valid_mask]
    tpr_sel = tpr[valid_mask]
    sorted_idx = np.argsort(fpr_sel)
    fpr_partial = fpr_sel[sorted_idx]
    tpr_partial = tpr_sel[sorted_idx]

    partial_auc = np.trapz(tpr_partial, fpr_partial)

    if normalize:
        fpr_range = fpr_partial[-1] - fpr_partial[0]
        if fpr_range > 0:
            partial_auc /= fpr_range
        else:
            partial_auc = 0.0

    return partial_auc, None, None, None


def roc(y_true, y_prob):
    """
    y_true: binary ground truth as a NumPy array (shape: (N,), values 0/1 or False/True)
    y_prob: probabilistic predictions as a NumPy array (shape: (N,), values in [0, 1])
    out_path: where to store the ROC arrays for later visualization
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # Basic sanity checks
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    if np.isnan(y_true).any() or np.isnan(y_prob).any():
        raise ValueError("Inputs contain NaNs.")
    if not np.array_equal(np.unique(y_true), np.intersect1d(np.unique(y_true), [0, 1])):
        # Convert booleans to {0,1} if needed, otherwise error
        if y_true.dtype == bool:
            y_true = y_true.astype(int)
        else:
            raise ValueError("y_true must be binary (0/1 or bool).")
    if (y_prob < 0).any() or (y_prob > 1).any():
        raise ValueError("y_prob must be probabilities in [0, 1].")

    # Compute ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    return fpr.tolist(), tpr.tolist(), thresholds.tolist(), float(roc_auc)


def sensitivity_specificity(y_true, y_pred):
    """
    Compute sensitivity and specificity from binary ground truth and binary predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary array (0/1 or bool).
    y_pred : np.ndarray
        Predicted binary array (0/1 or bool), same shape as y_true.

    Returns
    -------
    sensitivity : float
    specificity : float
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch between y_true and y_pred.")

    # Confusion matrix components
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan

    return sensitivity, specificity


def evaluate_tumor_model(preop_dir: Path, followup_dir: Path, pred_file: Path, model_id: str, ctv_margin: int = 15, csf_mask: bool = False) -> Dict[str, Any]:
    """
    Evaluate a tumor model by computing recurrence coverage for standard and 
    model-based radiotherapy plans using MRI segmentation data.

    Parameters:
        preop_dir (Path): Directory to the preoperative exam that has been preprocessed. Should contain the folder with the output.
        followup_dir (Path): Directory to the postoperative exam that has been preprocessed. Should contain the folder with the output.
        pred_file (Path): File path containing model prediction MRI data.
        model_id (str): Identifier for the model. Used for the name of the output file.
        ctv_margin (int, optional): Margin used to expand the clinical target volume for the standard plan in mm. Defaults to 15.
        csf_mask (bool, optional): If true, does not consider predictions/recurrences in CSF in any way by masking it out.

    Returns:
        Dict[str, Any]: Dictionary with computed metrics
    """
    results = {}

    # Load data
    brain_mask_dir = BRAIN_MASK_SCHEMA.format(base_dir=str(preop_dir))
    if brain_mask_dir.exists():
        brain_mask = np.rint(load_mri_data(str(brain_mask_dir))).astype(np.int32)
    else:
        t1c_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=str(preop_dir), modality="t1c")
        t1c = load_mri_data(str(t1c_file))
        background = np.min(t1c)
        brain_mask = np.rint(t1c > background)

    if csf_mask:
        tissue_segmentation_dir = TISSUE_SEG_SCHEMA.format(base_dir=str(preop_dir))
        tissue_segmentation = np.rint(load_mri_data(str(tissue_segmentation_dir))).astype(np.int32)
        brain_mask[tissue_segmentation==TISSUE_LABELS["csf"]] = 0

    tumorseg_dir = TUMORSEG_SCHEMA.format(base_dir=str(preop_dir))
    core_segmentation = np.rint(load_mri_data(str(tumorseg_dir))).astype(np.int32)
    core_segmentation[core_segmentation==2] = 0  # ignore edma
    core_segmentation[core_segmentation==3] = 1

    recurrence_dir = RECURRENCE_SCHEMA.format(base_dir=str(followup_dir))
    recurrence_segmentation = np.rint(load_mri_data(str(recurrence_dir))).astype(np.int32)
    recurrence_segmentation[recurrence_segmentation == 1] = 0  # ignore necrosis
    recurrence_segmentation[recurrence_segmentation == 2] = 0  # ignore edema
    recurrence_segmentation[recurrence_segmentation == 3] = 1
    recurrence_segmentation[recurrence_segmentation == 4] = 0  # ignore resection cavity 
    
    recurrence_segmentation_all = np.rint(load_mri_data(recurrence_dir)).astype(np.int32)
    recurrence_segmentation_all[recurrence_segmentation_all == 1] = 1
    recurrence_segmentation_all[recurrence_segmentation_all == 2] = 1
    recurrence_segmentation_all[recurrence_segmentation_all == 3] = 1
    recurrence_segmentation_all[recurrence_segmentation_all == 4] = 0

    model_prediction = load_and_resample_mri_data(str(pred_file), resample_params=core_segmentation.shape, interp_type=0)
    if is_binary_array(model_prediction):
        logger.info(f"Prediction {str(pred_file)} is binary. Generating distance fade for radiation planning.")
        model_prediction = generate_distance_fade_mask_no_plateau(model_prediction)

    # Create standard plan
    standard_plan = create_standard_plan(core_segmentation, ctv_margin)
    standard_plan[brain_mask==0] = 0
    standard_plan_volume = np.sum(standard_plan)
    standard_plan_coverage = recurrence_coverage(recurrence_segmentation, standard_plan)
    standard_plan_coverage_all = recurrence_coverage(recurrence_segmentation_all, standard_plan)

    # Create model based plan
    tumor_threshold = find_threshold(model_prediction, standard_plan_volume, initial_threshold=0.2)
    #tumor_threshold = 0.1
    model_plan = (model_prediction > tumor_threshold).astype(np.int32)
    #model_plan[brain_mask == 0] = 0
    model_recurrence_coverage = recurrence_coverage(recurrence_segmentation, model_plan)
    model_recurrence_coverage_all = recurrence_coverage(recurrence_segmentation_all, model_plan)

    # ROC AUC
    distance_fade_core = generate_distance_fade_mask_no_plateau(core_segmentation)
    fpr_model, tpr_model, thresholds, auc_model = roc(recurrence_segmentation, model_prediction)
    fpr_standard, tpr_standard, thresholds, auc_standard = roc(recurrence_segmentation, standard_plan)
    fpr_standard_fade, tpr_standard_fade, thresholds, auc_standard_fade = roc(recurrence_segmentation, distance_fade_core)

    #peritumoral_mask = create_standard_plan(core_segmentation, 1)
    #rec_seg = np.rint(load_mri_data(str(recurrence_dir))).astype(np.int32)
    #auc_model, fpr, tpr, thresholds = roc_auc(pred=model_prediction, seg=rec_seg, mask=brain_mask)
    #auc_model, fpr, tpr, thresholds = partial_roc_auc(pred=model_prediction, seg=rec_seg, mask=brain_mask)

    #auc_standard_fade, fpr, tpr, thresholds = roc_auc(pred=distance_fade_core, seg=rec_seg, mask=brain_mask)
    #auc_standard, fpr, tpr, thresholds = roc_auc(pred=standard_plan, seg=rec_seg, mask=brain_mask)
    #auc_standard_fade, fpr, tpr, thresholds = partial_roc_auc(pred=distance_fade_core, seg=rec_seg, mask=brain_mask)
    #auc_standard, fpr, tpr, thresholds = partial_roc_auc(pred=standard_plan, seg=rec_seg, mask=brain_mask)


    #imgtest = nib.Nifti1Image(distance_fade_core, np.eye(4))
    #nib.save(imgtest, "/home/home/lucas/tmp/test.nii.gz")

    # Save plans
    outfile_standard = STANDARD_PLAN_SCHEMA.format(base_dir=str(preop_dir))
    outfile_model = MODEL_PLAN_SCHEMA.format(base_dir=str(preop_dir), algo_id=model_id)

    outfile_standard.parent.mkdir(parents=True, exist_ok=True)
    outfile_model.parent.mkdir(parents=True, exist_ok=True)
    
    standard_plan_nifti = nib.Nifti1Image(standard_plan, affine=np.eye(4))
    nib.save(standard_plan_nifti, outfile_standard)

    model_img = nib.Nifti1Image(model_plan, affine=np.eye(4))
    nib.save(model_img, outfile_model)

    # Compute metrics
    results["recurrence_coverage_standard"] = standard_plan_coverage
    results["recurrence_coverage_standard_all"] = standard_plan_coverage_all
    results["recurrence_coverage_model"] = model_recurrence_coverage
    results["recurrence_coverage_model_all"] = model_recurrence_coverage_all
    
    results["missed_voxels_standard"] = missed_voxels(recurrence_segmentation, standard_plan)
    results["missed_voxels_standard_all"] = missed_voxels(recurrence_segmentation_all, standard_plan)
    results["missed_voxels_model"] = missed_voxels(recurrence_segmentation, model_plan)
    results["missed_voxels_model_all"] = missed_voxels(recurrence_segmentation_all, model_plan)

    results["specificity_model"] = specificity(recurrence_segmentation, model_plan)
    results["sensitivity_model"] = sensitivity(recurrence_segmentation, model_plan)
    results["specificity_standard"] = specificity(recurrence_segmentation, standard_plan)
    results["sensitivity_standard"] = sensitivity(recurrence_segmentation, standard_plan)

    print(f"Specificity (model): {results['specificity_model']}")
    print(f"Sensitivity (model): {results['sensitivity_model']}")
    print(f"Specificity (standard): {results['specificity_standard']}")
    print(f"Sensitivity (standard): {results['sensitivity_standard']}")

    if auc_model != 0.0:
        results["roc_auc_model"] = auc_model
        results["tpr_model"] = tpr_model
        results["fpr_model"] = fpr_model
    if auc_standard != 0.0:
        results["roc_standard"] = auc_standard
        results["tpr_standard"] = tpr_standard
        results["fpr_standard"] = fpr_standard
    if auc_standard_fade != 0.0:
        results["roc_auc_standard_fade"] = auc_standard_fade
        results["tpr_standard_fade"] = tpr_standard_fade
        results["fpr_standard_fade"] = fpr_standard_fade

    # Save results
    save_file = METRICS_SCHEMA.format(base_dir=followup_dir, algo_id=model_id)
    with open(save_file, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Finished evaluation of {preop_dir}. Saved results to {save_file}.")

    print(tumorseg_dir)
    print(recurrence_dir)
    print(outfile_standard)
    return results


if __name__ == "__main__":
    # Example:
    # python gbm_bench/evaluation/evaluate.py -preop_dir test_data/exam1 -followup_dir test_data/exam3 -pred_file test_data/exam1/processed/growth_models/sbtc/sbtc_pred.nii.gz
    # python gbm_bench/evaluation/evaluate.py -preop_dir /mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0024/0 -followup_dir /mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0024/2 -pred_file /mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0024/0/processed/growth_models/sbtc/sbtc_pred.nii.gz
    # python gbm_bench/evaluation/evaluate.py -preop_dir /mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0001/0 -followup_dir /mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0001/2 -pred_file /mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0001/0/processed/growth_models/sbtc/sbtc_pred.nii.gz
    parser = argparse.ArgumentParser()
    parser.add_argument("-preop_dir", type=str, help="Path.")
    parser.add_argument("-followup_dir", type=str, help="Path.")
    parser.add_argument("-pred_file", type=str, help="Algorithm identifier, should be the same as the folder for the algorithm in patient/exam/processed/.")
    args = parser.parse_args()

    results = evaluate_tumor_model(
            preop_dir=Path(args.preop_dir),
            followup_dir=Path(args.followup_dir),
            pred_file=Path(args.pred_file),
            model_id="sbtc"  # "test"
            )

    print(results)
