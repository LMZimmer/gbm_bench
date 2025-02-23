import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gbm_bench.utils.utils import compute_center_of_mass, load_mri_data, load_and_resample_mri_data, merge_pdfs


MODALITY_ORDER = ["t1c", "t1", "t2", "flair"]


def get_image_dirs(preprocessing_dir: str) -> Dict:
    MODALITY_ORDER = ["t1c", "t1", "t2", "flair"]
    image_dirs = {
            "stripped": [os.path.join(preprocessing_dir, "skull_stripped", m+"_bet_normalized.nii.gz") for m in MODALITY_ORDER],
            "tumorseg": os.path.join(preprocessing_dir, "tumor_segmentation/tumor_seg.nii.gz"),
            "tissueseg": [
                os.path.join(preprocessing_dir, "tissue_segmentation/tissue_seg.nii.gz"),
                os.path.join(preprocessing_dir, "tissue_segmentation/wm_pbmap.nii.gz"),
                os.path.join(preprocessing_dir, "tissue_segmentation/gm_pbmap.nii.gz"),
                os.path.join(preprocessing_dir, "tissue_segmentation/csf_pbmap.nii.gz")
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
    classes_of_interest: List[int] = [1, 2, 3]
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

    # colormap for tumor segmentation (1: non enhancing, 2: edema, 3: enhancing)
    colors = [(0,0,0,0), (1, 127/255, 0, 1), (30/255, 144/255, 1, 1), (138/255, 43/255, 226/255, 1)]
    color_labels = ["Non-enhancing Tumor", "Peritumoral Edema", "Enhancing Tumor"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    patches = [mpatches.Patch(color=c, label=l) for (c, l) in zip(colors[1:], color_labels)]

    # each key-value pair in image_dirs gets 2 rows (axial, sagittal)
    num_sequences = len(image_dirs["stripped"])
    num_rows = len(image_dirs)
    fig, axs = plt.subplots(2 * num_rows, num_sequences, figsize=(20, 8 * num_rows))

    # Threshold for tumor cell concentration in tumor model outputs
    c_threshold = 0.01

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
        overlay = np.where(overlay < c_threshold, 0, overlay)
        im = axs[3, i].imshow(overlay, cmap='inferno',  alpha=0.9, vmin=0.0, vmax=1.0)
        divider = make_axes_locatable(axs[3, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
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
        axs[7, i].imshow(overlay, cmap=cmap, norm=norm, alpha=0.9)
        axs[7, i].axis("off")

        # skull strippped + tumor model
        axs[8, i].imshow(np.rot90(load_mri_data(image_dirs["stripped"][i])[slice_num_sagittal, :, :]), cmap="gray")
        overlay = np.rot90(load_and_resample_mri_data(image_dirs["lmi"], resample_params=patient_dim, interp_type=1)[slice_num_sagittal, :, :])
        overlay = np.where(overlay < c_threshold, 0, overlay)
        axs[8, i].imshow(overlay, cmap='inferno', alpha=0.9, vmin=0.0, vmax=1.0)
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
            (
                f"Patient: {patient_identifier}\n"
                f"Exam: {exam_identifier}\n"
                f"Algorithm: {algorithm_identifier}\n"
                f"CoM slice (axial/sagittal): {slice_num_axial}/{slice_num_sagittal}\n"
                f"Tumor cell concentration threshold: {c_threshold}"
                ),
            horizontalalignment="left",
            fontsize=20,
            fontweight="bold",
            color="black",
            y=0.92,
            x=0.066,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

    # Color legends
    fig.legend(handles=patches, loc="upper right", bbox_to_anchor=(0.96, 0.891), ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(outfile, format="pdf")
    print(f"Plot saved as {outfile}")
    plt.close(fig)


def plot_tumor_concentration_multislice(
    patient_identifier: str,
    exam_identifier: str,
    algorithm_identifier: str,
    preprocessing_dir: str,
    outfile: str,
    classes_of_interest: List[int] = [1, 2, 3]
) -> None:

    image_dirs = get_image_dirs(preprocessing_dir)

    t1c_data = load_mri_data(image_dirs["stripped"][0])
    seg_data = load_mri_data(image_dirs["tumorseg"])
    patient_dim = t1c_data.shape

    # Compute center of mass of the tumor/region from the segmentation mask
    center = compute_center_of_mass(seg_data, t1c_data, classes_of_interest)
    step_size = 10
    num_slices = 5
    axial_slices = [center[2] + ind * step_size - 2 * step_size for ind in range(0, num_slices)]
    axial_slices = [min(max(0, ax_slice), patient_dim[2]) for ax_slice in axial_slices]
    sagittal_slices = [center[0] + ind * step_size - 2 * step_size for ind in range(0, num_slices)]
    sagittal_slices = [min(max(0, sag_slice), patient_dim[0]) for sag_slice in sagittal_slices]

    print(
        f"Computed center of mass (Axial slice: {center[2]}, Sagittal slice: {center[0]})"
    )

    # colormap for tumor segmentation (1: non enhancing, 2: edema, 3: enhancing)
    colors = [(0,0,0,0), (1, 127/255, 0, 1), (30/255, 144/255, 1, 1), (138/255, 43/255, 226/255, 1)]
    color_labels = ["Non-enhancing Tumor", "Peritumoral Edema", "Enhancing Tumor"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    patches = [mpatches.Patch(color=c, label=l) for (c, l) in zip(colors[1:], color_labels)]

    # each key-value pair in image_dirs gets 2 rows (axial, sagittal)
    fig, axs = plt.subplots(2 * num_slices, 4, figsize=(20, 8 * num_slices))

    # Threshold for tumor cell concentration in tumor model outputs
    c_threshold = 0.01

    # axial plots
    for i, axial_slice in enumerate(axial_slices):

        # T1c
        axs[i, 0].imshow(np.rot90(t1c_data[:, :, axial_slice]), cmap="gray")
        axs[i, 0].axis("off")

        # Tumor segmentation
        axs[i, 1].imshow(np.rot90(t1c_data[:, :, axial_slice]), cmap="gray")
        overlay = np.rot90(seg_data[:, :, axial_slice])
        axs[i, 1].imshow(overlay, cmap=cmap, norm=norm, alpha=0.9)
        axs[i, 1].axis("off")

        # Model
        axs[i, 2].imshow(np.rot90(t1c_data[:, :, axial_slice]), cmap="gray")
        overlay = np.rot90(load_and_resample_mri_data(image_dirs["lmi"], resample_params=patient_dim, interp_type=1)[:, :, axial_slice])
        overlay = np.where(overlay < c_threshold, 0, overlay)
        im = axs[i, 2].imshow(overlay, cmap='inferno',  alpha=0.90, vmin=0.0, vmax=1.0)
        divider = make_axes_locatable(axs[i, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        axs[i, 2].axis("off")

        # Tissue segmentation
        axs[i, 3].imshow(np.rot90(load_mri_data(image_dirs["tissueseg"][0])[:, :, axial_slice]), cmap="gray")
        axs[i, 3].axis("off")

    for i, sagittal_slice in enumerate(sagittal_slices):
        
        # T1c
        axs[i + num_slices, 0].imshow(np.rot90(t1c_data[sagittal_slice, :, :]), cmap="gray")
        axs[i + num_slices, 0].axis("off")

        # Tumor segmentation
        axs[i + num_slices, 1].imshow(np.rot90(t1c_data[sagittal_slice, :, :]), cmap="gray")
        overlay = np.rot90(seg_data[sagittal_slice, :, :])
        axs[i + num_slices, 1].imshow(overlay, cmap=cmap, norm=norm, alpha=0.9)
        axs[i + num_slices, 1].axis("off")

        # Model
        axs[i + num_slices, 2].imshow(np.rot90(t1c_data[sagittal_slice, :, :]), cmap="gray")
        overlay = np.rot90(load_and_resample_mri_data(image_dirs["lmi"], resample_params=patient_dim, interp_type=1)[sagittal_slice, :, :])
        overlay = np.where(overlay < c_threshold, 0, overlay)
        axs[i + num_slices, 2].imshow(overlay, cmap='inferno',  alpha=0.90, vmin=0.0, vmax=1.0)
        axs[i + num_slices, 2].axis("off")

        # Tissue segmentation
        axs[i + num_slices, 3].imshow(np.rot90(load_mri_data(image_dirs["tissueseg"][0])[sagittal_slice, :, :]), cmap="gray")
        axs[i + num_slices, 3].axis("off")

    # Column titles
    axs[0, 0].set_title("T1C", fontsize=16, fontweight="bold", pad=20)
    axs[0, 1].set_title("TUMORSEG", fontsize=16, fontweight="bold", pad=20)
    axs[0, 2].set_title(f"{algorithm_identifier.upper()}", fontsize=16, fontweight="bold", pad=20)
    axs[0, 3].set_title("TISSUESEG", fontsize=16, fontweight="bold", pad=20)

    # Row titles
    row_labels = axial_slices + sagittal_slices
    for ind, rl in enumerate(row_labels):
        axs[ind, 0].axis("on")
        axs[ind, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axs[ind, 0].set_ylabel(rl, fontweight="bold", labelpad=20, fontsize=16)

    # Add identifiers with adjusted margins and bounding box
    fig.subplots_adjust(top=0.85)
    fig.suptitle(
            (
                f"Patient: {patient_identifier}\n"
                f"Exam: {exam_identifier}\n"
                f"Algorithm: {algorithm_identifier}\n"
                f"CoM slice (axial/sagittal): {center[2]}/{center[0]}\n"
                f"Tumor cell concentration threshold: {c_threshold}"
                ),
            horizontalalignment="left",
            fontsize=20,
            fontweight="bold",
            color="black",
            y=0.92,
            x=0.066,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

    # Color legends
    fig.legend(handles=patches, loc="upper right", bbox_to_anchor=(0.96, 0.891), ncol=3)

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

    #plot_mri_with_segmentation(
    #        patient_identifier=args.patient_id,
    #        exam_identifier=args.exam_id,
    #        algorithm_identifier=args.algo_id,
    #        preprocessing_dir=args.preprocessing_dir,
    #        outfile=args.outfile
    #        )

    plot_tumor_concentration_multislice(
            patient_identifier=args.patient_id,
            exam_identifier=args.exam_id,
            algorithm_identifier=args.algo_id,
            preprocessing_dir=args.preprocessing_dir,
            outfile=args.outfile
            )
