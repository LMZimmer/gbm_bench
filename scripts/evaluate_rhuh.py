import os
import shutil
import pickle
import argparse
from gbm_bench.utils.utils import merge_pdfs
from gbm_bench.utils.parsing import RHUHParser
from gbm_bench.evaluation.evaluate import evaluate_tumor_model
from gbm_bench.utils.visualization import plot_model_multislice, plot_recurrence_multislice


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
    
    for ind, patient in enumerate(patients):
        
        print(f"Creating plots {ind}/{len(patients)}...")
        
        patient_identifier = patient["patient_id"]
        exam_identifier_preop = patient["exam_ids"][0]     # First exam is pre-op
        exam_identifier_followup = patient["exam_ids"][2]  # Second exam is post-op, Third is follow up
        preprocessing_dir_preop = os.path.join(patient["exams"][0], "preprocessing")
        preprocessing_dir_followup = os.path.join(patient["exams"][2], "preprocessing")
        algorithm_identifier = "lmi"                       # LMI, SBTC

        results = evaluate_tumor_model(
                preop_exam_dir=patient["exams"][0],
                postop_exam_dir=patient["exams"][2],
                algo_id=algorithm_identifier
                )

        outdir = os.path.join(preprocessing_dir_preop, "lmi/results.json")
        with open(outdir, "wb") as fp:
            pickle.dump(results, fp)

        print(f"{patient_identifier}: {results}")


        
