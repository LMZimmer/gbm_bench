import os
import argparse
import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt
from gbm_bench.utils.metrics import coverage
from gbm_bench.utils.utils import load_mri_data, load_and_resample_mri_data


def create_standard_plan(core_segmentation, ctv_margin):
    distance_transform = distance_transform_edt(~ (core_segmentation >0))
    dilated_core = distance_transform <= ctv_margin
    return dilated_core


def find_threshold(volume, target_volume, tolerance=0.01, initial_threshold=0.5,maxIter = 10000):

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


def getRecurrenceCoverage(tumorRecurrence, treatmentPlan):

    if np.sum(tumorRecurrence) <=  0.00001:
        return 1

    # Calculate the intersection between the recurrence and the plan
    intersection = np.logical_and(tumorRecurrence, treatmentPlan)

    # Calculate the coverage as the ratio of the intersection to the recurrence
    coverage = np.sum(intersection) / np.sum(tumorRecurrence)
    return coverage


def getPredictionInRecurrence(tumorRecurrence, treatmentPlan):

    if np.sum(tumorRecurrence) <=  0.00001:
        return 0

    if np.sum(treatmentPlan) <= 0.00001:
        return 0

    # normalize sum of treatment plan to 1
    normalizedTreatmentPlan = treatmentPlan / np.sum(treatmentPlan)

    coverage = np.sum((tumorRecurrence > 0) * normalizedTreatmentPlan)

    return coverage


def evaluate_tumor_model(preop_exam_dir, postop_exam_dir, algo_id, ctv_margin=15, tumor_conc_thresh=0.1):
    
    results = {}

    # Load data
    brain_msak_dir = os.path.join(preop_exam_dir, "preprocessing/skull_stripped/t1c_bet_mask.nii.gz")
    brain_mask = load_mri_data(brain_msak_dir)

    core_segmentation_dir = os.path.join(preop_exam_dir, "preprocessing/tumor_segmentation/enhancing_non_enhancing_tumor.nii.gz")
    core_segmentation = load_mri_data(core_segmentation_dir)

    recurrence_dir = os.path.join(postop_exam_dir, "preprocessing/longitudinal/recurrence_preop.nii.gz")
    recurrence_segmentation = load_mri_data(recurrence_dir)
    recurrence_segmentation[recurrence_segmentation == 2] = 0    # ignore edema

    model_prediction_dir = os.path.join(preop_exam_dir, f"preprocessing/{algo_id}/lmi_tumor_patientSpace.nii")
    model_prediction = load_and_resample_mri_data(model_prediction_dir, resample_params=core_segmentation.shape, interp_type=0)

    #tumor_cell_density_thresh = np.where(tumor_cell_density < tumor_conc_thresh, 0, tumor_cell_density)
    #threshold_based_plan = create_standard_plan(np.where(tumor_cell_density_thresh, ctv_margin)


    # Create conventional plan
    conventional_plan = create_standard_plan(core_segmentation, ctv_margin)
    conventional_plan[brain_mask == 0] = 0
    conventional_plan_volume = np.sum(conventional_plan)
    conventional_plan_coverage = getRecurrenceCoverage(recurrence_segmentation, conventional_plan)
    #conventional_plan_coverage_all = getRecurrenceCoverage(recurrenceAll, standardPlan)

    # Create model based plan
    tumor_threshold = find_threshold(model_prediction, conventional_plan_volume, initial_threshold=0.2)
    model_recurrence_coverage = getRecurrenceCoverage(recurrence_segmentation, model_prediction > tumor_threshold)
    #model_recurrence_coverage_all = getRecurrenceCoverage(recurrenceAll, model_prediction > tumor_threshold)

    # Compute metrics
    results["recurrence_coverage_conventional"] = conventional_plan_coverage
    results["recurrence_coverage_model"] = model_recurrence_coverage
    return results


if __name__ == "__main__":
    #python gbm_bench/evaluation/evaluate.py -preop_exam_dir test_data/exam1 -postop_exam_dir test_data/exam3 -algo_id lmi
    parser = argparse.ArgumentParser()
    parser.add_argument("-preop_exam_dir", type=str, help="Path.")
    parser.add_argument("-postop_exam_dir", type=str, help="Path.")
    parser.add_argument("-algo_id", type=str, help="Algorithm identifier, should be the same as the folder for the algorithm in patient/exam/preprocessing/.")
    args = parser.parse_args()

    results = evaluate_tumor_model(
            preop_exam_dir=args.preop_exam_dir,
            postop_exam_dir=args.postop_exam_dir,
            algo_id=args.algo_id
            )

    print(results)
