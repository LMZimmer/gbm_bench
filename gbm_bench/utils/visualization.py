import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple
from gbm_bench.utils.utils import compute_center_of_mass, load_mri_data, load_and_resample_mri_data, merge_pdfs


MODALITY_ORDER = ["t1c", "t1", "t2", "flair"]


def get_image_dirs(preprocessing_dir: str) -> Dict:
    MODALITY_ORDER = ["t1c", "t1", "t2", "flair"]
    image_dirs = {
            "stripped": [os.path.join(preprocessing_dir, "skull_stripped", m+"_bet_normalized.nii.gz") for m in MODALITY_ORDER],
            "tumorseg": os.path.join(preprocessing_dir, "tumor_segmentation/tumor_seg.nii.gz"),
            "tissueseg": [
                os.path.join(preprocessing_dir, "tissue_segmentation/tissue_seg.nii.gz"),
                os.path.join(preprocessing_dir, "tissue_segmentation/wm.nii.gz"),
                os.path.join(preprocessing_dir, "tissue_segmentation/gm.nii.gz"),
                os.path.join(preprocessing_dir, "tissue_segmentation/csf.nii.gz")
                ],
            "lmi": os.path.join(preprocessing_dir, "lmi/lmi_tumor_patientSpace.nii"),
            "masks": [
                os.path.join(preprocessing_dir, "skull_stripped/t1c_bet_mask.nii.gz"),
                os.path.join(preprocessing_dir, "tumor_segmentation/tumor_seg.nii.gz"),
                os.path.join(preprocessing_dir, "tumor_segmentation/enhancing_non_enhancing_tumor.nii.gz"),
                os.path.join(preprocessing_dir, "tumor_segmentation/peritumoral_edema.nii.gz")
                ]
            }
    return image_dirs


def plot_mri_with_segmentation(
    patient_identifier: str,
    exam_identifier: str,
    algorithm_identifier: str,
    preprocessing_dir: str,
    outfile: str,
    classes_of_interest: List[int] = [1, 2, 3],
) -> None:
    
    image_dirs = get_image_dirs(preprocessing_dir)

    t1c_data = load_mri_data(image_dirs["stripped"][0])
    seg_data = load_mri_data(image_dirs["tumorseg"])
    patient_dim = t1c_data.shape

    # Compute center of mass of the tumor/region from the segmentation mask
    center = compute_center_of_mass(seg_data, t1c_data, classes_of_interest)
    slice_num_axial = center[2]     # z-axis
    slice_num_sagittal = center[0]  # x-axis

    print(
        f"Computed center of mass (Axial slice: {slice_num_axial}, Sagittal slice: {slice_num_sagittal})"
    )

    # colormap for tumor segmentation (none for background)
    colors = ["none", "green", "blue", "red"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [i - 0.5 for i in classes_of_interest] + [classes_of_interest[-1] + 0.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # each key-value pair in image_dirs gets 2 rows (axial, sagittal)
    num_sequences = len(image_dirs["stripped"])
    num_rows = len(image_dirs)
    fig, axs = plt.subplots(2 * num_rows, num_sequences, figsize=(20, 8 * num_rows))

    # axial plots
    for i, modality in enumerate(MODALITY_ORDER):

        # skull stripped
        axs[0, i].imshow(np.rot90(load_mri_data(image_dirs["stripped"][i])[:, :, slice_num_axial]), cmap="gray")
        axs[0, i].set_title(modality.upper(), fontsize=16, fontweight="bold", pad=20)
        axs[0, i].axis("off")

        # tissue segmentation
        axs[1, i].imshow(np.rot90(load_mri_data(image_dirs["tissueseg"][i])[:, :, slice_num_axial]), cmap="gray")
        axs[1, i].axis("off")

        # skull stripped + tumor segmentation
        axs[2, i].imshow(np.rot90(load_mri_data(image_dirs["stripped"][i])[:, :, slice_num_axial]), cmap="gray")
        overlay = np.rot90(seg_data[:, :, slice_num_axial])
        axs[2, i].imshow(overlay, cmap=cmap, norm=norm, alpha=0.9)
        axs[2, i].axis("off")

        # skull strippped + tumor model
        axs[3, i].imshow(np.rot90(load_mri_data(image_dirs["stripped"][i])[:, :, slice_num_axial]), cmap="gray")
        overlay = np.rot90(load_and_resample_mri_data(image_dirs["lmi"], resample_params=patient_dim, interp_type=1)[:, :, slice_num_axial])
        #overlay = np.rot90(load_mri_data(image_dirs["lmi"])[:, :, slice_num_axial])
        axs[3, i].imshow(overlay, cmap='inferno',  alpha=0.8)
        axs[3, i].axis("off")

        # masks
        axs[4, i].imshow(np.rot90(load_mri_data(image_dirs["masks"][i])[:, :, slice_num_axial]), cmap="gray")
        axs[4, i].axis("off")

    # Sagittal plots
    for i, modality in enumerate(MODALITY_ORDER):
        
        # skull stripped
        axs[5, i].imshow(np.rot90(load_mri_data(image_dirs["stripped"][i])[slice_num_sagittal, :, :]), cmap="gray")
        axs[5, i].axis("off")

        # tissue segmentation
        axs[6, i].imshow(np.rot90(load_mri_data(image_dirs["tissueseg"][i])[slice_num_sagittal, :, :]), cmap="gray")
        axs[6, i].axis("off")

        # skull stripped + tumor segmentation
        axs[7, i].imshow(np.rot90(load_mri_data(image_dirs["stripped"][i])[slice_num_sagittal, :, :]), cmap="gray")
        overlay = np.rot90(seg_data[slice_num_sagittal, :, :])
        axs[7, i].imshow(overlay, cmap=cmap, norm=norm, alpha=0.8)
        axs[7, i].axis("off")

        # skull strippped + tumor model
        axs[8, i].imshow(np.rot90(load_mri_data(image_dirs["stripped"][i])[slice_num_sagittal, :, :]), cmap="gray")
        overlay = np.rot90(load_and_resample_mri_data(image_dirs["lmi"], resample_params=patient_dim, interp_type=1)[slice_num_sagittal, :, :])
        #overlay = np.rot90(load_mri_data(image_dirs["lmi"])[slice_num_sagittal, :, :])
        axs[8, i].imshow(overlay, cmap='inferno', alpha=0.9)
        axs[8, i].axis("off")

        # masks
        axs[9, i].imshow(np.rot90(load_mri_data(image_dirs["masks"][i])[slice_num_sagittal, :, :]), cmap="gray")
        axs[9, i].axis("off")

    # Left hand side titles
    row_labels = ["STRIPPED", "TISSUESEG", "TUMORSEG", "MODEL", "MASKS"]
    for ind, rl in enumerate(row_labels):
        axs[ind, 0].axis("on")
        axs[ind, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axs[ind, 0].set_ylabel(rl, fontweight="bold", labelpad=20, fontsize=16)
        axs[ind+len(row_labels), 0].axis("on")
        axs[ind+len(row_labels), 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axs[ind+len(row_labels), 0].set_ylabel(rl, fontweight="bold", labelpad=20, fontsize=16)

    # Add identifiers with adjusted margins and bounding box
    fig.subplots_adjust(top=0.85)
    fig.suptitle(
            f"Patient: {patient_identifier}\nExam: {exam_identifier}\nAlgorithm: {algorithm_identifier}\nSlice (axial/sagittal): {slice_num_axial}/{slice_num_sagittal}",
        fontsize=20,
        fontweight="bold",
        color="black",
        y=0.95,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(outfile, format="pdf")
    print(f"Plot saved as {outfile}")
    plt.close(fig)


if __name__ == "__main__":
    # Example:
    # python gbm_bench/utils/visualization.py -preprocessing_dir /home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0001/01-25-2015-NA-RM\ CEREBRAL6NEURNAV-21029/preprocessing/ -patient_id RHUH-0001 -exam_id 01-25-2015 -algo_id LMI -outfile ~/test.pdf
    parser = argparse.ArgumentParser()
    parser.add_argument("-preprocessing_dir", type=str, help="Directory containing outputs. Should be named 'preprocessing'.")
    parser.add_argument("-patient_id", type=str, help="Patient identifier for plot.")
    parser.add_argument("-exam_id", type=str, help="Exam identifier for plot.")
    parser.add_argument("-algo_id", type=str, help="Algorithm identifier for plot.")
    parser.add_argument("-outfile", type=str, help="Directory to save figure to.")
    args = parser.parse_args()

    plot_mri_with_segmentation(
            patient_identifier=args.patient_id,
            exam_identifier=args.exam_id,
            algorithm_identifier=args.algo_id,
            preprocessing_dir=args.preprocessing_dir,
            outfile=args.outfile
            )
