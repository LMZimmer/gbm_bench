import os
import shutil
import pickle
import argparse
import numpy as np
from scipy import stats
from pathlib import Path
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.evaluation.evaluate import evaluate_tumor_model
from gbm_bench.utils.constants import PREDICTION_OUTPUT_SCHEMA


if __name__ == "__main__":
    # Example:
    # python scripts/evaluate_gliodil10.py -algorithm sbtc
    # nohup python -u scripts/evaluate_gliodil10.py -algorithm sbtc > gliodil_sbtc.txt 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-algorithm", type=str, help="Algorithm ID to evaluate.")
    args = parser.parse_args()
    
    # Read dataset
    gliodil_root = "/mnt/Drive2/lucas/datasets/GLIODIL"
    gliodil = LongitudinalDataset(dataset_id="GLIODIL10", root_dir=gliodil_root)
    GLIODIL_DIR = Path("/home/home/lucas/projects/gbm_bench/gbm_bench/data/datasets/gliodil10.json") 
    gliodil.load(GLIODIL_DIR)

    print(f"Evaluating {args.algorithm}")
    all_results = []

    for patient_ind, patient in enumerate(gliodil.patients):
        print(f"Processing {patient_ind}/{len(gliodil.patients)}...")

        patient_identifier = patient["patient_id"]
        preop_exams = gliodil.get_patient_exams(patient_id=patient_identifier, timepoint="preop")
        followup_exams = gliodil.get_patient_exams(patient_id=patient_identifier, timepoint="followup")

        if len(preop_exams) > 1:
            print(f"Warning: found {len(preop_exams)} preop exams for patient {patiend_ind, patiend}. Using first exam for evaluation.")

        algo_id = args.algorithm
        preop_exam_dir = preop_exams[0]["tumorseg"].parent / "preop"
        prediction_dir = PREDICTION_OUTPUT_SCHEMA.format(base_dir=preop_exam_dir, algo_id=algo_id)
        
        for followup_exam in followup_exams:
            followup_exam_dir = followup_exam["tumorseg"].parent / "followup"

            try:
                results = evaluate_tumor_model(
                        preop_dir=preop_exam_dir,
                        followup_dir=followup_exam_dir,
                        pred_file=prediction_dir,
                        model_id=algo_id
                        )
                all_results.append(results)
                print(f"{patient_identifier}: {results}")
            except Exception as e:
                raise e
                print(f"Exception for {followup_exam_dir}: {e}")

    recurrence_coverage_standard = [r["recurrence_coverage_standard"] for r in all_results]
    recurrence_coverage_standard_all = [r["recurrence_coverage_standard_all"] for r in all_results]
    recurrence_coverage_model = [r["recurrence_coverage_model"] for r in all_results]
    recurrence_coverage_model_all = [r["recurrence_coverage_model_all"] for r in all_results]

    print(f"Finished evaluation.")
    print(f"Standard plan coverge: {100*np.mean(recurrence_coverage_standard):.2f} \u00B1 {100*stats.sem(recurrence_coverage_standard):.2f}")
    print(f"Standard plan coverge (all): {100*np.mean(recurrence_coverage_standard_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_standard_all):.2f}")
    print(f"Model plan coverge: {100*np.mean(recurrence_coverage_model):.2f} \u00B1 {100*stats.sem(recurrence_coverage_model):.2f}")
    print(f"Model plan coverge (all): {100*np.mean(recurrence_coverage_model_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_model_all):.2f}")

