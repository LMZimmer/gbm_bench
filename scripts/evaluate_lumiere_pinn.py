import os
import shutil
import pickle
import argparse
import numpy as np
from scipy import stats
from pathlib import Path
from gbm_bench.utils.constants import LUMIERE_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.evaluation.evaluate import evaluate_tumor_model
from gbm_bench.utils.constants import PREDICTION_OUTPUT_SCHEMA


if __name__ == "__main__":
    # Example:
    # nohup python -u scripts/evaluate_lumiere_pinn.py > evaluate_pinngbm_lumiere.txt 2>&1 &
    
    # Read dataset
    lumiere_root = "/mnt/Drive2/lucas/datasets/LUMIERE/Imaging"
    lumiere = LongitudinalDataset(dataset_id="LUMIERE", root_dir=lumiere_root)
    lumiere.load(LUMIERE_DIR)

    pinndir = Path("/mnt/Drive2/ray/predict_eval/")

    all_results = []

    for patient_ind, patient in enumerate(lumiere.patients):
        print(f"Processing {patient_ind}/{len(lumiere.patients)}...")

        patient_identifier = patient["patient_id"]
        preop_exams = lumiere.get_patient_exams(patient_id=patient_identifier, timepoint="preop")
        followup_exams = lumiere.get_patient_exams(patient_id=patient_identifier, timepoint="followup")

        if len(preop_exams) > 1:
            print(f"Warning: found {len(preop_exams)} preop exams for patient {patiend_ind, patient_identifier}. Using first exam for evaluation.")

        preop_exam_dir = preop_exams[0]["t1"].parent
        #prediction_dir = PREDICTION_OUTPUT_SCHEMA.format(base_dir=preop_exam_dir, algo_id=algo_id)
        prediction_dir = pinndir / patient_identifier / "pinngbm_pred.nii.gz"
        
        for followup_exam in followup_exams:
            followup_exam_dir = followup_exam["t1"].parent

            try:
                results = evaluate_tumor_model(
                        preop_dir=preop_exam_dir,
                        followup_dir=followup_exam_dir,
                        pred_file=prediction_dir,
                        model_id="pinngbm"
                        )
                all_results.append(results)
            except Exception as e:
                print(e)

    recurrence_coverage_standard = [r["recurrence_coverage_standard"] for r in all_results]
    recurrence_coverage_standard_all = [r["recurrence_coverage_standard_all"] for r in all_results]
    recurrence_coverage_model = [r["recurrence_coverage_model"] for r in all_results]
    recurrence_coverage_model_all = [r["recurrence_coverage_model_all"] for r in all_results]

    print(f"Finished evaluation.")
    print(f"Standard plan coverge: {100*np.mean(recurrence_coverage_standard):.2f} \u00B1 {100*stats.sem(recurrence_coverage_standard):.2f}")
    print(f"Standard plan coverge (all): {100*np.mean(recurrence_coverage_standard_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_standard_all):.2f}")
    print(f"Model plan coverge: {100*np.mean(recurrence_coverage_model):.2f} \u00B1 {100*stats.sem(recurrence_coverage_model):.2f}")
    print(f"Model plan coverge (all): {100*np.mean(recurrence_coverage_model_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_model_all):.2f}")

