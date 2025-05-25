import os
import shutil
import argparse
from pathlib import Path
from gbm_bench.utils.utils import merge_pdfs
from gbm_bench.utils.constants import RHUH_GBM_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.utils.visualization import plot_full_brain


if __name__ == "__main__":
    # Example:
    # python scripts/sanity_check.py

    # Read dataset
    rhuh_root = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
    rhuh_gbm = LongitudinalDataset(dataset_id="RHUH", root_dir=rhuh_root)
    rhuh_gbm.load(RHUH_GBM_DIR)

    outfiles = []
    tmp_dir = "./tmp/sanity"
    os.makedirs(tmp_dir, exist_ok=True)
    
    for patient_ind, patient in enumerate(rhuh_gbm.patients):
        print(f"Visualizing {patient_ind}/{len(rhuh_gbm.patients)}...")
        
        patient_identifier = patient["patient_id"]
        exam_dir_preopop = rhuh_gbm.get_patient_exams(patient_id=patient_identifier, timepoint="preop")[0]["t1c"].parent
        exam_dir_followup = rhuh_gbm.get_patient_exams(patient_id=patient_identifier, timepoint="followup")[0]["t1c"].parent
        exam_identifier_preop = str(exam_dir_preopop.name)
        exam_identifier_followup = str(exam_dir_followup.name)

        algorithm_identifier = "sbtc"                       # LMI, SBTC, GLIODIL
        
        outfile = os.path.join(tmp_dir, f"{patient_identifier}_sanity.pdf")
        outfiles.append(outfile)
        
        plot_full_brain(
            patient_identifier=patient_identifier,
            exam_identifier_pre=exam_identifier_preop,
            exam_identifier_followup=exam_identifier_followup,
            exam_dir_preop=exam_dir_preopop,
            exam_dir_followup=exam_dir_followup,
            outfile=outfile
            )

    # Merge PDFs
    outfiles.sort()
    merge_pdfs(outfiles, f"./tmp/RHUH_sanity.pdf")

    # Delete temporary files
    shutil.rmtree(tmp_dir)
