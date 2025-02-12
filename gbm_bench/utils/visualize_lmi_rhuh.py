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


def load_mri_data(filepath: str) -> np.ndarray:
    img = nib.load(filepath)
    data = img.get_fdata()
    return data


def compute_center_of_mass(
    seg_data: np.ndarray,
    mri_data: np.ndarray,
    classes: List[int] = [1, 2, 3],
) -> Tuple[int, int, int]:
    mask = np.isin(seg_data, classes)

    # Check if the mask contains any non-zero values (i.e., non-empty segmentation)
    if not np.any(mask):
        print("Warning: Segmentation is empty, returning middle slices of the MRI.")
        # Return the middle slices of the MRI volume as default
        return (mri_data.shape[0] // 2, mri_data.shape[1] // 2, mri_data.shape[2] // 2)

    # Compute center of mass if the segmentation is non-empty
    com = center_of_mass(mask)
    return tuple(map(int, com))


def load_mri_data(filepath: str) -> np.ndarray:
    img = nib.load(filepath)
    data = img.get_fdata()
    return data


def compute_center_of_mass(
    seg_data: np.ndarray, mri_data: np.ndarray, classes: List[int] = [1, 2, 3]
) -> Tuple[int, int, int]:
    mask = np.isin(seg_data, classes)

    if not np.any(mask):
        print("Warning: Segmentation is empty, returning middle slices of the MRI.")
        return mri_data.shape[0] // 2, mri_data.shape[1] // 2, mri_data.shape[2] // 2

    com = center_of_mass(mask)
    return tuple(map(int, com))


def plot_mri_with_segmentation(
    patient_identifier: str,
    exam_identifier: str,
    algorithm_identifier: str,
    t1c_path: str,
    t1_path: str,
    t2_path: str,
    fla_path: str,
    seg_path: str,
    output_pdf: str,
    classes_of_interest: List[int] = [1, 2, 3],
) -> None:
    # Load the MRI sequences
    t1c_data = load_mri_data(t1c_path)
    t1_data = load_mri_data(t1_path)
    t2_data = load_mri_data(t2_path)
    fla_data = load_mri_data(fla_path)

    # Load segmentation mask
    seg_data = load_mri_data(seg_path)

    # Compute center of mass of the tumor/region from the segmentation mask
    center = compute_center_of_mass(seg_data, t1c_data, classes_of_interest)
    slice_num_axial = center[2]  # z-axis
    slice_num_sagittal = center[0]  # x-axis

    print(
        f"Computed center of mass (Axial slice: {slice_num_axial}, Sagittal slice: {slice_num_sagittal})"
    )

    # Updated sequence order: T1c, T1, T2, FLAIR
    sequences = [
        ("T1c", t1c_data),
        ("T1", t1_data),
        ("T2", t2_data),
        ("FLAIR", fla_data),
    ]

    # Set up the plot with 4 rows of images per sequence
    num_sequences = len(sequences)
    fig, axs = plt.subplots(4, num_sequences, figsize=(20, 16))

    # Plot axial images (raw and with overlays)
    for i, (seq_name, data) in enumerate(sequences):
        # Axial images (raw)
        axs[0, i].imshow(np.rot90(data[:, :, slice_num_axial]), cmap="gray")
        axs[0, i].set_title(f"{seq_name} - Axial (Raw): {slice_num_axial}")
        axs[0, i].axis("off")

        # Axial view with segmentation overlay (only non-zero regions)
        axs[1, i].imshow(np.rot90(data[:, :, slice_num_axial]), cmap="gray")
        overlay = np.rot90(seg_data[:, :, slice_num_axial])
        axs[1, i].imshow(overlay, cmap='viridis', alpha=0.9)
        axs[1, i].set_title(f"{seq_name} - Axial (Overlay): {slice_num_axial}")
        axs[1, i].axis("off")

    # Plot sagittal images (raw and with overlays)
    for i, (seq_name, data) in enumerate(sequences):
        # Sagittal images (raw)
        axs[2, i].imshow(np.rot90(data[slice_num_sagittal, :, :]), cmap="gray")
        axs[2, i].set_title(f"{seq_name} - Sagittal (Raw): {slice_num_sagittal}")
        axs[2, i].axis("off")

        # Sagittal view with segmentation overlay (only non-zero regions)
        axs[3, i].imshow(np.rot90(data[slice_num_sagittal, :, :]), cmap="gray")
        overlay = np.rot90(seg_data[slice_num_sagittal, :, :])
        axs[3, i].imshow(overlay, cmap='viridis', alpha=0.9)
        axs[3, i].set_title(f"{seq_name} - Sagittal (Overlay): {slice_num_sagittal}")
        axs[3, i].axis("off")

    # Add identifiers with adjusted margins and bounding box
    fig.subplots_adjust(top=0.85)  # Increase top margin to fit text
    fig.suptitle(
        f"Patient: {patient_identifier}\nExam: {exam_identifier}\nAlgorithm: {algorithm_identifier}",
        fontsize=16,
        fontweight="bold",
        color="black",
        y=0.95,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to fit identifiers

    # Save the figure as a PDF to the custom output location
    plt.savefig(output_pdf, format="pdf")
    print(f"Plot saved as {output_pdf}")

    # Close the figure to free up memory
    plt.close(fig)


def plot_exam(
    patient_identifier: str,
    exam_identifier: str,
    algorithm_identifier: str,
    exam_path: str,
    output_pdf: str,
    classes_of_interest: List[int] = [1, 2, 3],
) -> None:
    exam_path = turbopath(exam_path)

    t1c_path = os.path.join(exam_path, "preprocessing/skull_stripped/t1c_bet_normalized.nii.gz")
    t1_path = os.path.join(exam_path, "preprocessing/skull_stripped/t1_bet_normalized.nii.gz")
    t2_path = os.path.join(exam_path, "preprocessing/skull_stripped/t2_bet_normalized.nii.gz")
    fla_path = os.path.join(exam_path, "preprocessing/skull_stripped/flair_bet_normalized.nii.gz")

    seg_path = os.path.join(exam_path, "preprocessing/lmi/lmi_tumor_patientSpace.nii")

    # Plot with computed center of mass slices and save to the specified PDF
    plot_mri_with_segmentation(
        patient_identifier=patient_identifier,
        exam_identifier=exam_identifier,
        algorithm_identifier=algorithm_identifier,
        t1c_path=t1c_path,
        t1_path=t1_path,
        t2_path=t2_path,
        fla_path=fla_path,
        seg_path=seg_path,
        output_pdf=output_pdf,
        classes_of_interest=classes_of_interest,
    )


import re
from pathlib import Path
from PyPDF2 import PdfMerger
from typing import List
import os
from tqdm import tqdm


def extract_day_number(name: str) -> int:
    match = re.search(r"\d+", name)
    return int(match.group()) if match else 0


def merge_pdfs(pdf_list: List[str], output_pdf: str) -> None:
    pdf_merger = PdfMerger()

    for pdf in pdf_list:
        pdf_merger.append(pdf)

    pdf_merger.write(output_pdf)
    pdf_merger.close()
    print(f"Combined PDF saved as {output_pdf}")


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


if __name__ == "__main__":
    # Example:
    # python visualize_exam_rhuh.py

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
