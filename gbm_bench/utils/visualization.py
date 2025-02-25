import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import colormaps
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
            x=0.0665,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

    # Color legends
    fig.legend(handles=patches, loc="upper right", bbox_to_anchor=(0.96, 0.891), ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(outfile, format="pdf")
    print(f"Plot saved as {outfile}")
    plt.close(fig)


def plot_recurrence(
    patient_identifier: str,
    exam_identifier_pre: str,
    exam_identifier_post: str,
    preprocessing_dir_pre: str,
    preprocessing_dir_post: str,
    outfile: str,
    classes_of_interest: List[int] = [1, 2, 3]
) -> None:

    t1c_pre_dir = os.path.join(preprocessing_dir_pre, "skull_stripped/t1c_bet_normalized.nii.gz")
    t1c_post_dir = os.path.join(preprocessing_dir_post, "longitudinal/t1c_warped_longitudinal.nii.gz")
    tumor_seg_dir = os.path.join(preprocessing_dir_pre, "tumor_segmentation/tumor_seg.nii.gz")
    recurrence_seg_dir = os.path.join(preprocessing_dir_post, "longitudinal/recurrence_preop.nii.gz")

    t1c_data_pre = load_mri_data(t1c_pre_dir)
    seg_data_pre = load_mri_data(tumor_seg_dir)
    t1c_data_post = load_mri_data(t1c_post_dir)
    seg_data_post = load_mri_data(recurrence_seg_dir)
    patient_dim = t1c_data_pre.shape

    # Compute center of mass of the tumor/region from the segmentation mask
    center = compute_center_of_mass(seg_data_pre, t1c_data_pre, classes_of_interest)
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

    # axial plots
    for i, axial_slice in enumerate(axial_slices):

        # T1c (pre)
        axs[i, 0].imshow(np.rot90(t1c_data_pre[:, :, axial_slice]), cmap="gray")
        axs[i, 0].axis("off")

        # T1c (pre) + Tumor seg (pre)
        axs[i, 1].imshow(np.rot90(t1c_data_pre[:, :, axial_slice]), cmap="gray")
        overlay = np.rot90(seg_data_pre[:, :, axial_slice])
        axs[i, 1].imshow(overlay, cmap=cmap, norm=norm, alpha=0.9)
        axs[i, 1].axis("off")

        # T1c (post)
        axs[i, 2].imshow(np.rot90(t1c_data_post[:, :, axial_slice]), cmap="gray")
        axs[i, 2].axis("off")

        # T1c (post) + Tumor seg (recurrence)
        axs[i, 3].imshow(np.rot90(t1c_data_post[:, :, axial_slice]), cmap="gray")
        overlay = np.rot90(seg_data_post[:, :, axial_slice])
        axs[i, 3].imshow(overlay, cmap=cmap, norm=norm, alpha=0.9)
        axs[i, 3].axis("off")

    for i, sagittal_slice in enumerate(sagittal_slices):

        # T1c (pre)
        axs[i + num_slices, 0].imshow(np.rot90(t1c_data_pre[sagittal_slice, :, :]), cmap="gray")
        axs[i + num_slices, 0].axis("off")

        # T1c (pre) + Tumor seg (pre)
        axs[i + num_slices, 1].imshow(np.rot90(t1c_data_pre[sagittal_slice, :, :]), cmap="gray")
        overlay = np.rot90(seg_data_pre[sagittal_slice, :, :])
        axs[i + num_slices, 1].imshow(overlay, cmap=cmap, norm=norm, alpha=0.9)
        axs[i + num_slices, 1].axis("off")

        # T1c (post)
        axs[i + num_slices, 2].imshow(np.rot90(t1c_data_post[sagittal_slice, :, :]), cmap="gray")
        axs[i + num_slices, 2].axis("off")

        # T1c (post) + Tumor seg (recurrence)
        axs[i + num_slices, 3].imshow(np.rot90(t1c_data_post[sagittal_slice, :, :]), cmap="gray")
        overlay = np.rot90(seg_data_post[sagittal_slice, :, :])
        axs[i + num_slices, 3].imshow(overlay, cmap=cmap, norm=norm, alpha=0.9)
        axs[i + num_slices, 3].axis("off")

    # Column titles
    axs[0, 0].set_title("T1C (preop)", fontsize=16, fontweight="bold", pad=20)
    axs[0, 1].set_title("T1C (preop)+Tumor", fontsize=16, fontweight="bold", pad=20)
    axs[0, 2].set_title("T1C (postop)", fontsize=16, fontweight="bold", pad=20)
    axs[0, 3].set_title("T1C (postop)+Recurrence", fontsize=16, fontweight="bold", pad=20)

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
                f"Exam (preop): {exam_identifier_pre}\n"
                f"Exam (postop): {exam_identifier_post}\n"
                f"CoM slice (axial/sagittal): {center[2]}/{center[0]}\n"
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
































def grid_plot(
        image_tensor: np.ndarray,
        imshow_args: List,
        header: str,
        col_titles: List,
        row_titles: List,
        outfile: str,
        legend_handles: List[mpatches.Patch] = None
) -> None:
    """
    A generic function to create a grid plot with multiple layers / overlays.

    Args:
        image_tensor: A numpy array with dimension 3 (n_layers, n_cols, n_rows) where each point is a 2D-image or None.
            Note that due to this data type, all images are required to have the same dimension.
        imshow_args: A list of arguments for each image layer as taken by imshow.
        header: String to be displayd at the top of the image.
        col_titles: List of strings used as column titles.
        row_titles: List of strings used as row titles.
        outfile: File that the image is saved to.
        legend_handles: List of matplotlib.patches.Patch to be displayed in a legend.
    """

    if image_tensor.ndim != 3:
        raise ValueError("Dimension mismatch. image_tensor dimension should be 5: (n_layers, n_cols, n_rows)")

    if len(imshow_args) != image_tensor.shape[0]:
        raise ValueError(f"Dimension mismatch. imshow_args should be the same length as image_tensor.shape[0] = {image_tensor.shape[0]}.")

    if len(row_titles) != image_tensor.shape[1]:
        raise ValueError(f"Dimension mismatch. row_titles should be the same length as image_tensor.shape[1] = {image_tensor.shape[1]}.")

    if len(col_titles) != image_tensor.shape[2]:
        raise ValueError(f"Dimension mismatch. col_titles should be the same length as image_tensor.shape[2] = {image_tensor.shape[2]}.")

    n_row = image_tensor.shape[1]
    n_col = image_tensor.shape[2]
    non_gray_cmaps = [mpcmp for mpcmp in colormaps() if mpcmp not in ["grey", "gray"]]

    # Axes
    fig, axs = plt.subplots(n_row, n_col, figsize=(5 * n_col, 4 * n_row))
    for image_layer, imshow_args in zip(image_tensor, imshow_args):
        for row in range(n_row):
            for col in range(n_col):
                if image_layer[row, col] is not None:
                    img = axs[row, col].imshow(np.rot90(image_layer[row, col]), **imshow_args)
                    axs[row, col].axis("off")

                    # Add colorbar if non gray colormap is used
                    if "cmap" in imshow_args.keys() and imshow_args["cmap"] in non_gray_cmaps:
                        divider = make_axes_locatable(axs[row, col])
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(img, cax=cax)

    # Column titles
    for ind, col_title in enumerate(col_titles):
        axs[0, ind].set_title(col_title, fontsize=16, fontweight="bold", pad=20)

    # Row titles
    for ind, row_title in enumerate(row_titles):
        axs[ind, 0].axis("on")
        axs[ind, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axs[ind, 0].set_ylabel(row_title, fontweight="bold", labelpad=20, fontsize=16)

    # Header
    fig.subplots_adjust(top=0.85)
    fig.suptitle(
            header,
            horizontalalignment="left",
            fontsize=20,
            fontweight="bold",
            color="black",
            y=0.92,
            x=0.0665,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

    # Color legends
    if legend_handles is not None:
        fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.96, 0.890), ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(outfile, format="pdf")
    print(f"Plot saved as {outfile}")
    plt.close(fig)


def plot_model_multislice(
    patient_identifier: str,
    exam_identifier: str,
    algorithm_identifier: str,
    preprocessing_dir: str,
    outfile: str
) -> None:

    c_threshold = 0.01    # tumor cell concentration threshold
    n_layers = 3    # one layer for each imshow config
    classes_of_interest = [1, 2, 3]    # classes in tumor seg

    # Load data
    image_dirs = get_image_dirs(preprocessing_dir)
    t1c_data = load_mri_data(image_dirs["stripped"][0])
    tumorseg_data = load_mri_data(image_dirs["tumorseg"])
    tissueseg_data = load_mri_data(image_dirs["tissueseg"][0])
    model_data = load_and_resample_mri_data(image_dirs["lmi"], resample_params=t1c_data.shape, interp_type=1)

    # Comput tumor center of mass
    center = compute_center_of_mass(tumorseg_data, t1c_data, classes_of_interest)

    # Create axial/coronal slices
    step_size = 10
    num_slices = 5
    patient_dim = t1c_data.shape
    axial_slices = [center[2] + ind * step_size - 2 * step_size for ind in range(0, num_slices)]
    axial_slices = [min(max(0, ax_slice), patient_dim[2]) for ax_slice in axial_slices]
    coronal_slices = [center[1] + ind * step_size - 2 * step_size for ind in range(0, num_slices)]
    coronal_slices = [min(max(0, cor_slice), patient_dim[1]) for cor_slice in coronal_slices]

    # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
    classes_of_interest = [1, 2, 3]
    colors = [(0,0,0,0), (1, 127/255, 0, 1), (30/255, 144/255, 1, 1), (138/255, 43/255, 226/255, 1)]
    color_labels = ["Non-enhancing Tumor", "Peritumoral Edema", "Enhancing Tumor"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    patches = [mpatches.Patch(color=c, label=l) for (c, l) in zip(colors[1:], color_labels)]

    # Titles
    col_titles = ["T1C", "TUMORSEG", f"{algorithm_identifier.upper()}", "TISSUESEG"]
    row_titles = axial_slices + coronal_slices
    header = (
            f"Patient: {patient_identifier}\n"
            f"Exam: {exam_identifier}\n"
            f"Algorithm: {algorithm_identifier}\n"
            f"CoM slice (axial/coronal): {center[2]}/{center[1]}\n"
            f"Tumor cell concentration threshold: {c_threshold}"
            )

    # Build image tensor
    image_tensor = np.empty((n_layers, num_slices*2, 4), dtype=object)
    
    # Layer 1: T1c, T1c, T1c, Tissueseg
    layer_1_args = {"cmap": "gray"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        
        image_tensor[0, ind, 0] = t1c_data[:, :, ax_slice]
        image_tensor[0, ind, 1] = t1c_data[:, :, ax_slice]
        image_tensor[0, ind, 2] = t1c_data[:, :, ax_slice]
        image_tensor[0, ind, 3] = tissueseg_data[:, :, ax_slice]

        image_tensor[0, ind+num_slices, 0] = t1c_data[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 1] = t1c_data[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 2] = t1c_data[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 3] = tissueseg_data[:, cor_slice, :]

    # Layer 2: None, Tumorseg, None, None
    layer_2_args = {"cmap": cmap, "norm": norm, "alpha": 0.9}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        
        image_tensor[1, ind, 1] = tumorseg_data[:, :, ax_slice]
        image_tensor[1, ind+num_slices, 1] = tumorseg_data[:, cor_slice, :]

    # Layer 3: None, None, Model, None
    layer_3_args = {"cmap": "inferno", "alpha": 0.90, "vmin": 0.0, "vmax": 1.0}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        
        image_tensor[2, ind, 2] = model_data[:, :, ax_slice]
        image_tensor[2, ind+num_slices, 2] = model_data[:, cor_slice, :]

    # Imshow arguments
    imshow_args = [layer_1_args, layer_2_args, layer_3_args]

    grid_plot(
            image_tensor=image_tensor,
            imshow_args=imshow_args,
            header=header,
            col_titles=col_titles,
            row_titles=row_titles,
            outfile=outfile,
            legend_handles=patches
            )


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

    #plot_tumor_concentration_multislice(
    #        patient_identifier=args.patient_id,
    #        exam_identifier=args.exam_id,
    #        algorithm_identifier=args.algo_id,
    #        preprocessing_dir=args.preprocessing_dir,
    #        outfile=args.outfile
    #        )

    #plot_recurrence(
    #        patient_identifier="RHUH-0001",
    #        exam_identifier_pre="Pre",
    #        exam_identifier_post="Post",
    #        preprocessing_dir_pre="test_data/exam1/preprocessing",
    #        preprocessing_dir_post="test_data/exam3/preprocessing",
    #        outfile="test_longitudinal.pdf"
    #        )

    plot_model_multislice(
            patient_identifier="RHUH-0001",
            exam_identifier="Pre",
            algorithm_identifier="lmi",
            preprocessing_dir="test_data/exam1/preprocessing",
            outfile="test_newplot.pdf"
            )
