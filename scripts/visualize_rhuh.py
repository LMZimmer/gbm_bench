import os
import shutil
import argparse
from gbm_bench.utils.utils import merge_pdfs
from gbm_bench.utils.parsing import RHUHParser
from gbm_bench.utils.visualization import plot_model_multislice, plot_recurrence_multislice


if __name__ == "__main__":
    # Example:
    # python scripts/visualize_rhuh.py

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
        algorithm_identifier = "SBTC"                       # LMI, SBTC
        
        outfile_model = os.path.join(tmp_dir_model, f"{patient_identifier}_{algorithm_identifier}.pdf")
        outfiles_model.append(outfile_model)

        plot_model_multislice(
                patient_identifier=patient_identifier,
                exam_identifier=exam_identifier_preop,
                algorithm_identifier=algorithm_identifier,
                preprocessing_dir=preprocessing_dir_preop,
                outfile=outfile_model
                )

        outfile_recurrence = os.path.join(tmp_dir_rec, f"{patient_identifier}_recurrence.pdf")
        outfiles_recurrences.append(outfile_recurrence)
        
        #plot_recurrence_multislice(
        #    patient_identifier=patient_identifier,
        #    exam_identifier_pre=exam_identifier_preop,
        #    exam_identifier_post=exam_identifier_followup,
        #    preprocessing_dir_pre=preprocessing_dir_preop,
        #    preprocessing_dir_post=preprocessing_dir_followup,
        #    outfile=outfile_recurrence
        #    )

    # Merge PDFs
    outfiles_model.sort(), outfiles_recurrences.sort()
    merge_pdfs(outfiles_model, f"./tmp/RHUH_{algorithm_identifier}.pdf")
    #merge_pdfs(outfiles_recurrences, f"./tmp/RHUH_recurrences.pdf")

    # Delete temporary files
    shutil.rmtree(tmp_dir_model)
    shutil.rmtree(tmp_dir_rec)
