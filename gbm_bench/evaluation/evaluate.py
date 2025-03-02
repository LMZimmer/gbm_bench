import os
import argparse
from scipy.ndimage import center_of_mass, distance_transform_edt
from gbm_bench.utils.metrics import coverage
from gbm_bench.utils.utils import load_mri_data, load_and_resample_mri_data


def create_standard_plan(core_segmentation, distance):
    distance_transform = distance_transform_edt(~ (core_segmentation >0))
    dilated_core = distance_transform <= distance
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

    # Create conventional plan
    core_segmentation_dir = os.path.join(preop_exam_dir, "preprocessing/tumor_segmentation/enhancing_non_enhancing_tumor.nii.gz")
    core_segmentation = load_mri_data(core_segmentation_dir)
    conventional_plan = create_standard_plan(core_segmentation, ctv_margin)

    #standardPlan = tools.create_standard_plan(tumorCore, wandb.config.standardPlanDistance)
    #standardPlan[brainMaskNumpy == False] = 0
    #standardPlanVolume = np.sum(standardPlan)
    #standardRecurrencePlanCoverage = tools.getRecurrenceCoverage(recurrenceCore, standardPlan)
    #standardRecurrencePlanCoverageAll = tools.getRecurrenceCoverage(recurrenceAll, standardPlan)


    # Create model based plan
    tumor_cell_density_dir = os.path.join(postop_exam_dir, f"{algo_id}/lmi_tumor_patientSpace.nii")
    tumor_cell_density = load_and_resample_mri_data(tumor_cell_density_dir, resample_params=core_segmentation.shape, interp_type=0)
    tumor_cell_density_thresh = np.where(tumor_cell_density < tumor_conc_thresh, 0, tumor_cell_density)
    threshold_based_plan = create_standard_plan(np.where(tumor_cell_density_thresh, ctv_margin)

    #TODO: How to create a fair rad plan? 1. concentration, 2. iso-volumetric
    #tumorThreshold = tools.find_threshold(tumorNumpy, standardPlanVolume, initial_threshold= tumorThreshold)
    #recurrenceCoverage = tools.getRecurrenceCoverage(recurrenceCore , tumorNumpy > tumorThreshold)
    #recurrenceCoverageAll = tools.getRecurrenceCoverage(recurrenceAll , tumorNumpy > tumorThreshold)

    # Compute metrics
    recurrence_segmentation_dir = os.path.join(postop_exam_dir, "preprocessing/tumor_segmentation/enhancing_non_enhancing_tumor.nii.gz")
    recurrence_segmentation = load_mri_data(recurrence_segmentation_dir)
    results["recurrence_coverage_conventional"] = coverage(recurrence_segmentation, conventional_plan)
    results[f"recurrence_coverage_{algo_id}"] = coverage(recurrence_segmentation, threshold_based_plan)
    results["rmse"] = None
    results["mse"] = None

    return results


if __name__ == "__main__":
    #python gbm_bench/evaluation/evaluate.py -preop_exam_dir test_data/exam1 -postop_exam_dir test_data/exam2 -algo_id lmi
    parser = argparse.ArgumentParser()
    parser.add_argument("-preop_exam_dir", type=str, help="Path.")
    parser.add_argument("-postop_exam_dir", type=str, help="Path.")
    parser.add_argument("-algo_id", type=str, help="Algorithm identifier, should be the same as the folder for the algorithm in patient/exam/preprocessing/.")
    args = parser.parse_args()

    evaluate_tumor_model(
            preop_exam_dir=args.preop_exam_dir,
            postop_exam_dir=args.postop_exam_dir,
            algo_id=args.algo_id
            )
