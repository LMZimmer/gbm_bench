import os
import shutil
import pickle
import argparse
import numpy as np
from gbm_bench.utils.utils import merge_pdfs
from gbm_bench.utils.parsing import RHUHParser
from gbm_bench.evaluation.evaluate import evaluate_tumor_model


if __name__ == "__main__":
    # Example:
    # python scripts/evaluate_rhuh.py

    os.environ["CUDA_VISIBLE_DEVICES"]="4"

    rhuh_root = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
    rhuh_parser = RHUHParser(root_dir=rhuh_root)
    rhuh_parser.parse()
    patients = rhuh_parser.get_patients()

    outfiles_model, outfiles_recurrences = [], []
    tmp_dir_model, tmp_dir_rec = "./tmp/model", "./tmp/recurrence"
    os.makedirs(tmp_dir_model, exist_ok=True)
    os.makedirs(tmp_dir_rec, exist_ok=True)

    all_results = []
    
    for ind, patient in enumerate(patients):
        
        print(f"Patient {ind}/{len(patients)}...")
        
        patient_identifier = patient["patient_id"]
        exam_identifier_preop = patient["exam_ids"][0]     # First exam is pre-op
        exam_identifier_followup = patient["exam_ids"][2]  # Second exam is post-op, Third is follow up
        preprocessing_dir_preop = os.path.join(patient["exams"][0], "preprocessing")
        preprocessing_dir_followup = os.path.join(patient["exams"][2], "preprocessing")
        #prediction_dir = os.path.join(preprocessing_dir_preop, "lmi/lmi_tumor_patientSpace.nii")
        #prediction_dir = os.path.join(preprocessing_dir_preop, "sbtc/recurrencePrediction.nii.gz")
        prediction_dir = os.path.join(preprocessing_dir_preop, "gliodilx_pet__PDE1.0_/192_48_48_48_solution.nii")

        try:
            results = evaluate_tumor_model(
                    preop_exam_dir=patient["exams"][0],
                    postop_exam_dir=patient["exams"][2],
                    pred_dir=prediction_dir
                    )

            outdir = os.path.join(os.path.dirname(prediction_dir), "coverage.pkl")
            pickle.dump(results, open(outdir, "wb"))
            all_results.append(results)
        except:
            print(f"Failed for {patient_identifier}. Possibly file not found. Continuing...")

        print(f"{patient_identifier}: {results}")

    recurrence_coverage_standard = [r["recurrence_coverage_standard"] for r in all_results]
    recurrence_coverage_standard_all = [r["recurrence_coverage_standard_all"] for r in all_results]
    recurrence_coverage_model = [r["recurrence_coverage_model"] for r in all_results]
    recurrence_coverage_model_all = [r["recurrence_coverage_model_all"] for r in all_results]

    print(f"Finished evaluation.")
    print(f"Standard plan coverge: {np.mean(recurrence_coverage_standard)} \u00B1 {np.std(recurrence_coverage_standard)}")
    print(f"Standard plan coverge (all): {np.mean(recurrence_coverage_standard_all)} \u00B1 {np.std(recurrence_coverage_standard_all)}")
    print(f"Model plan coverge: {np.mean(recurrence_coverage_model)} \u00B1 {np.std(recurrence_coverage_model)}")
    print(f"Model plan coverge (all): {np.mean(recurrence_coverage_model_all)} \u00B1 {np.std(recurrence_coverage_model_all)}")
    


        
