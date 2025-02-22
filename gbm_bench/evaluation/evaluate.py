import os
import argparse
from scipy.ndimage import center_of_mass, distance_transform_edt
from gbm_bench.utils.metrics import coverage
from gbm_bench.utils.utils import load_mri_data, load_and_resample_mri_data


def create_standard_plan(core_segmentation, distance):
    distance_transform = distance_transform_edt(~ (core_segmentation >0))
    dilated_core = distance_transform <= distance
    return dilated_core


def evaluate_tumor_model(preop_exam_dir, postop_exam_dir, algo_id, ctv_margin=15, tumor_conc_thresh=0.1):
    
    results = {}

    # Create conventional plan
    core_segmentation_dir = os.path.join(preop_exam_dir, "preprocessing/tumor_segmentation/enhancing_non_enhancing_tumor.nii.gz")
    core_segmentation = load_mri_data(core_segmentation_dir)
    conventional_plan = create_standard_plan(core_segmentation, ctv_margin)

    # Create model based plan
    tumor_cell_density_dir = os.path.join(postop_exam_dir, f"{algo_id}/lmi_tumor_patientSpace.nii")
    tumor_cell_density = load_and_resample_mri_data(tumor_cell_density_dir, resample_params=core_segmentation.shape, interp_type=0)
    tumor_cell_density_thresh = np.where(tumor_cell_density < tumor_conc_thresh, 0, tumor_cell_density)
    threshold_based_plan = create_standard_plan(np.where(tumor_cell_density_thresh, ctv_margin)

    #TODO: How to create a fair rad plan? 1. concentration, 2. iso-volumetric

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
