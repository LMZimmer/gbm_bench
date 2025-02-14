import glob
import os
import shutil
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from PyPDF2 import PdfMerger
from typing import List, Tuple
from scipy.ndimage import center_of_mass
from auxiliary.turbopath.turbopath import turbopath


"""
def rhuh_sort_func(exam_dir):
    date = os.path.basename(exam_dir).replace("-", "")
    date = date[4:8] + date[0:2] + date[2:4]
    return date


def rhuh_parse_exams(patient_dir, preop):
    # Parse patients
    patients = glob.glob(os.path.join(patient_dir, "RHUH-*"))
    # Parse exams, sort by date, return first for preop and later exams for postop
    exams = []
    for p in patients:
        patient_exams = glob.glob(os.path.join(p, "*-NA-*"))
        patient_exams.sort(key=rhuh_sort_func)
        if preop:
            exams.append(patient_exams[0])
        else:
            exams = exams + patient_exams[1:]
    return exams
"""


if __name__ == "__main__":
    # Example:
    # python visualize_exam_rhuh.py

    os.environ["CUDA_VISIBLE_DEVICES"]="7"

    # Loop over data and algorithms
    data_folder = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
    preop_exams = rhuh_parse_exams(data_folder, preop=True)
    clear_old_visualization = False

    patient_exams = []
    for exam in preop_exams:
        print(f"{exam}")
        patient_identifier = exam.split("/")[-2]
        exam_identifier = "0"
        algorithm_identifier = "BRATS"
        output_pdf = os.path.join(exam, f"preprocessing/visualization/{algorithm_identifier}_{patient_identifier}_{exam_identifier}.pdf")
        
        if clear_old_visualization:
            print(f"Clearing old visualizations in {os.path.dirname(output_pdf)}")
            shutil.rmtree(os.path.dirname(output_pdf))
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

        plot_exam(
                patient_identifier=patient_identifier,
                exam_identifier=exam_identifier,
                algorithm_identifier=algorithm_identifier,
                exam_path=exam,
                output_pdf=output_pdf,
                )
        patient_exams.append(output_pdf)

        # Merge all PDFs for this algorithm into one for the patient
        #combined_pdf_path = f"{patient_report_folder}/combined/{algorithm}_{patient.name}_combined.pdf"
        #merge_pdfs(patient_exams, combined_pdf_path)


    # THIS WAS LMI VISUALIZATION
    os.environ["CUDA_VISIBLE_DEVICES"]="7"

    # Loop over data and algorithms
    data_folder = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
    preop_exams = rhuh_parse_exams(data_folder, preop=True)

    patient_exams = []
    for exam in preop_exams:
        print(f"{exam}")
        patient_identifier = exam.split("/")[-2]
        exam_identifier = "0"
        algorithm_identifier = "LMI"
        output_pdf = os.path.join(exam, f"preprocessing/visualization/{algorithm_identifier}_{patient_identifier}_{exam_identifier}.pdf")
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

        plot_exam(
                patient_identifier=patient_identifier,
                exam_identifier=exam_identifier,
                algorithm_identifier=algorithm_identifier,
                exam_path=exam,
                output_pdf=output_pdf,
                )
        patient_exams.append(output_pdf)

        # Merge all PDFs for this algorithm into one for the patient
        #combined_pdf_path = f"{patient_report_folder}/combined/{algorithm}_{patient.name}_combined.pdf"
        #merge_pdfs(patient_exams, combined_pdf_path)
