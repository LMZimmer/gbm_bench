import os
import json
import math
import pickle
import random
import argparse
import statistics as stats
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.stats import median_abs_deviation
from loguru import logger
from pathlib import Path
from matplotlib import colormaps
from typing import Dict, List, Union, Tuple
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.utils.utils import compute_center_of_mass, load_mri_data, load_and_resample_mri_data, merge_pdfs
from gbm_bench.utils.constants import (
        BRAIN_MASK_SCHEMA,
        LONGITUDINAL_WARP_SCHEMA,
        METRICS_SCHEMA,
        MODALITY_CONVERTED_SCHEMA,
        MODALITY_STRIPPED_SCHEMA,
        MODEL_PLAN_SCHEMA,
        PREDICTION_OUTPUT_SCHEMA,
        RECURRENCE_SCHEMA,
        STANDARD_PLAN_SCHEMA,
        TISSUE_SEG_SCHEMA,
        TISSUE_PBMAP_SCHEMA,
        TUMORSEG_CORE_SCHEMA,
        TUMORSEG_SCHEMA
        )


NO_T1C = [
        "data_001",
        "data_013",
        "data_020",
        "data_030",
        "data_034",
        "data_991",
        "data_992",
        "data_994",
        "data_995",
        "data_998"
        ]


def get_slices(center: Tuple[int, int, int], num_slices: int, step_size: int, patient_dim: Tuple[int, int, int]):
    axial_slices = [center[2] + ind * step_size - 2 * step_size for ind in range(0, num_slices)]
    axial_slices = [min(max(0, ax_slice), patient_dim[2]-1) for ax_slice in axial_slices]
    coronal_slices = [center[1] + ind * step_size - 2 * step_size for ind in range(0, num_slices)]
    coronal_slices = [min(max(0, cor_slice), patient_dim[1]-1) for cor_slice in coronal_slices]
    return axial_slices, coronal_slices


def get_cmap_norm_patches_tumorseg(classes_of_interest: List[int]):
    # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
    colors = [(0,0,0,0), (1, 127/255, 0, 1), (30/255, 144/255, 1, 1), (138/255, 43/255, 226/255, 1)]
    color_labels = ["Non-enhancing Tumor", "Peritumoral Edema", "Enhancing Tumor"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    patches = [mpatches.Patch(color=c, label=l) for (c, l) in zip(colors[1:], color_labels)]
    return cmap, norm, patches


def get_cmap_norm_patches_tumorseg_5(classes_of_interest: List[int]):
    # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
    colors = [(0,0,0,0), (1, 127/255, 0, 1), (30/255, 144/255, 1, 1), (138/255, 43/255, 226/255, 1), (34/255., 139/255., 34/255., 1), (210/255., 43/255., 43/255., 1)]
    color_labels = ["Necrosis", "Peritumoral Edema", "Enhancing Tumor", "Standard Plan", "Model Plan"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    patches = [mpatches.Patch(color=c, label=l) for (c, l) in zip(colors[1:], color_labels)]
    return cmap, norm, patches


def get_segmentation_projection(segmentation: np.ndarray, label: int, axis: int) -> np.ndarray:
    seg_data = segmentation.copy()
    seg_data[seg_data!=label]=0
    projection = np.rint(np.sum(seg_data, axis=axis) > 0)
    return projection


def grid_plot(image_tensor: np.ndarray, imshow_args: List[Dict], header: str, col_titles: List[str], row_titles: List[str],
              outfile: str, legend_handles: List[mpatches.Patch] = None ) -> None:
    """
    A generic function to create a grid plot with multiple layers / overlays.

    Args:
        image_tensor: A numpy array with dimension 3 (n_layers, n_cols, n_rows) where each point is a 2D-image or None.
        imshow_args: A list of dictionaries containing arguments for imshow calls for each image layer (e.g. {"cmap": "gray"}).
        header: String to be displayd at the top of the image.
        col_titles: List of strings used as column titles.
        row_titles: List of strings used as row titles.
        outfile: File that the pdf is saved to.
        legend_handles: List of matplotlib.patches.Patch to be displayed in a legend.
    """

    if image_tensor.ndim != 3:
        raise ValueError("Dimension mismatch. image_tensor dimension should be 3: (n_layers, n_cols, n_rows)")

    if len(imshow_args) != image_tensor.shape[0]:
        raise ValueError(f"Dimension mismatch. imshow_args should be the same length as image_tensor.shape[0] = {image_tensor.shape[0]}.")

    if len(row_titles) != image_tensor.shape[1]:
        raise ValueError(f"Dimension mismatch. row_titles should be the same length as image_tensor.shape[1] = {image_tensor.shape[1]}.")

    #if len(col_titles) != image_tensor.shape[2]:
    #    raise ValueError(f"Dimension mismatch. col_titles should be the same length as image_tensor.shape[2] = {image_tensor.shape[2]}.")

    n_row = image_tensor.shape[1]
    n_col = image_tensor.shape[2]
    non_gray_cmaps = [mpcmp for mpcmp in colormaps() if mpcmp not in ["grey", "gray"]]

    # Create figure and fill axes
    fig, axs = plt.subplots(n_row, n_col, figsize=(5 * n_col, 4 * n_row))
    for image_layer, imshow_args in zip(image_tensor, imshow_args):
        for row in range(n_row):
            for col in range(n_col):
                if image_layer[row, col] is not None:
                    img = axs[row, col].imshow(np.rot90(image_layer[row, col]), **imshow_args)
                    if "alpha" in imshow_args.keys() and imshow_args["alpha"] < 0.4:  #TODO terrible implementation for time
                        from skimage.measure import find_contours
                        contours = find_contours(np.rot90(image_layer[row, col]), 0.5)
                        for c in contours:
                            axs[row, col].plot(c[:,1], c[:,0], linewidth=1, color=imshow_args["cmap"].colors[1])
                    axs[row, col].axis("off")

                    # Uncomment to add colorbar, increases vertical spacing
                    #if "cmap" in imshow_args.keys() and imshow_args["cmap"] in non_gray_cmaps:
                    #    divider = make_axes_locatable(axs[row, col])
                    #    cax = divider.append_axes("right", size="5%", pad=0.05)
                    #    plt.colorbar(img, cax=cax)

    if len(axs.flatten()) == len(col_titles):
        for ct, ax in zip(col_titles, axs.flatten()):
            ax.set_title(ct, fontsize=16, pad=20)
    else:
        for ind, col_title in enumerate(col_titles):
            axs[0, ind].set_title(col_title, fontsize=16, fontweight="bold", pad=20)

    # Row titles
    for ind, row_title in enumerate(row_titles):
        axs[ind, 0].axis("on")
        axs[ind, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axs[ind, 0].set_ylabel(row_title, fontweight="bold", labelpad=20, fontsize=16)

    # Header
    fig.subplots_adjust(wspace=0.0, hspace=0.25, top=0.95, bottom=0.05, left=0.1, right=0.9)
    #fig.subplots_adjust(top=0.85)
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
        #fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.96, 0.890), ncol=3)
        fig.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(0.8, 0.01), ncol=len(legend_handles))

    #plt.tight_layout(rect=[0, 0.05, 1.0, 0.9])
    #plt.tight_layout(rect=[0, 0.0, 1.0, 0.9])
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, format="pdf")
    print(f"Plot saved as {outfile}")
    plt.close(fig)


def plot_model_multislice(patient_identifier: str, exam_identifier: str, algorithm_identifier: str, exam_dir: Path,
                          outfile: str, classes_of_interest: List[int] = [1, 2, 3]) -> None:

    c_threshold = 0.01    # tumor cell concentration threshold
    n_layers = 3    # one layer for each imshow config

    # Load data
    t1c_data = load_mri_data(MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir, modality="t1c"))
    tumorseg_data = load_mri_data(TUMORSEG_SCHEMA.format(base_dir=exam_dir))
    tissueseg_data = load_mri_data(TISSUE_SEG_SCHEMA.format(base_dir=exam_dir))
    model_data = load_and_resample_mri_data(PREDICTION_OUTPUT_SCHEMA.format(base_dir=exam_dir, algo_id=algorithm_identifier.lower()), resample_params=t1c_data.shape, interp_type=1)

    # Mask data outside of the brain
    #NOTE: do we want this

    # Compute tumor center of mass
    center = compute_center_of_mass(tumorseg_data, t1c_data, classes_of_interest)

    # Create axial/coronal slices
    step_size = 10
    num_slices = 5
    patient_dim = t1c_data.shape
    axial_slices, coronal_slices = get_slices(center, num_slices, step_size, patient_dim)

    # Tumor segmentation args
    cmap, norm, patches = get_cmap_norm_patches_tumorseg(classes_of_interest)

    # Read recurrence coverage for title
    coverage_str = ""
    #coverage_dir = os.path.join(os.path.dirname(model_dir), "coverage.pkl") #TODO: update path
    #if os.path.isfile(coverage_dir):
    #    coverage = pickle.load(open(coverage_dir, "rb"))
    #    coverage_str = (
    #            f"Coverage (conventional / model): {100*coverage['recurrence_coverage_standard']:.1f}% / {100*coverage['recurrence_coverage_model']:.1f}%\n"
    #            f"CoverageAll (conventional / model): {100*coverage['recurrence_coverage_standard_all']:.1f}% / {100*coverage['recurrence_coverage_model_all']:.1f}%"
    #            )
    #else:
    #    coverage_str = ""

    # Titles
    col_titles = ["T1C", "TUMORSEG", f"{algorithm_identifier.upper()}", "TISSUESEG"]
    row_titles = axial_slices + coronal_slices
    header = (
            f"Patient: {patient_identifier}\n"
            f"Exam: {exam_identifier}\n"
            f"Algorithm: {algorithm_identifier}\n"
            f"Tumor cell concentration threshold: {c_threshold}\n" + coverage_str
            )

    # Build image tensor
    image_tensor = np.empty((n_layers, num_slices*2, 4), dtype=object)
    
    # Layer 1: T1c, T1c, T1c, Tissueseg
    layer_1_args = {"cmap": "gray", "interpolation": "none"}
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
    layer_2_args = {"cmap": cmap, "norm": norm, "alpha": 0.9, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        
        image_tensor[1, ind, 1] = tumorseg_data[:, :, ax_slice]
        image_tensor[1, ind+num_slices, 1] = tumorseg_data[:, cor_slice, :]

    # Layer 3: None, None, Model, None
    layer_3_args = {"cmap": "inferno", "alpha": 0.90, "vmin": 0.0, "vmax": 1.0, "interpolation": "none"}
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


def plot_recurrence_multislice(patient_identifier: str, exam_identifier_pre: str, exam_identifier_followup: str,
                               exam_dir_preop: Path, exam_dir_followup: Path, outfile: str,
                               classes_of_interest: List[int] = [1, 2, 3]) -> None:

    n_layers = 2    # one layer for each imshow config

    # Paths
    t1c_pre_dir = MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir_preop, modality="t1c")
    t1c_post_dir = LONGITUDINAL_WARP_SCHEMA.format(base_dir=exam_dir_followup)
    #t1c_post_dir = MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir_followup, modality="t1c")  # non-co-registered version
    tumor_seg_dir = TUMORSEG_SCHEMA.format(base_dir=exam_dir_preop)
    recurrence_seg_dir = RECURRENCE_SCHEMA.format(base_dir=exam_dir_followup)
    #recurrence_seg_dir = TUMORSEG_SCHEMA.format(base_dir=exam_dir_followup)  # non-co-registered version

    # Load images
    t1c_data_pre = load_mri_data(t1c_pre_dir)
    seg_data_pre = load_mri_data(tumor_seg_dir)
    t1c_data_post = load_mri_data(t1c_post_dir)
    seg_data_post = load_mri_data(recurrence_seg_dir)

    seg_data_post[seg_data_post==4] = 0  # ignore ressection cavity label

    # Compute tumor center of mass
    center = compute_center_of_mass(seg_data_pre, t1c_data_pre, classes_of_interest)

    # Create axial/coronal slices
    step_size = 10
    num_slices = 5
    patient_dim = t1c_data_pre.shape
    axial_slices, coronal_slices = get_slices(center, num_slices, step_size, patient_dim)

    # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
    cmap, norm, patches = get_cmap_norm_patches_tumorseg(classes_of_interest)

    # Titles
    col_titles = ["T1C (preop)", "T1C (preop)+Tumor", "T1C (follow up)", "T1C (follow up) + Recurrence"]
    row_titles = axial_slices + coronal_slices
    header = (
            f"Patient: {patient_identifier}\n"
            f"Exam (preop): {exam_identifier_pre}\n"
            f"Exam (follow up): {exam_identifier_followup}\n"
            f"CoM slice (axial/coronal): {center[2]}/{center[1]}\n"
            )

    # Build image tensor
    image_tensor = np.empty((n_layers, num_slices*2, 4), dtype=object)

    # Layer 1: T1c (pre), T1c (pre), T1c (post, T1c (post)
    layer_1_args = {"cmap": "gray", "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):

        image_tensor[0, ind, 0] = t1c_data_pre[:, :, ax_slice]
        image_tensor[0, ind, 1] = t1c_data_pre[:, :, ax_slice]
        image_tensor[0, ind, 2] = t1c_data_post[:, :, ax_slice]
        image_tensor[0, ind, 3] = t1c_data_post[:, :, ax_slice]

        image_tensor[0, ind+num_slices, 0] = t1c_data_pre[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 1] = t1c_data_pre[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 2] = t1c_data_post[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 3] = t1c_data_post[:, cor_slice, :]

    # Layer 2: None, Tumorseg (pre), None, Tumorseg (post)
    layer_2_args = {"cmap": cmap, "norm": norm, "alpha": 0.9, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):

        image_tensor[1, ind, 1] = seg_data_pre[:, :, ax_slice]
        image_tensor[1, ind, 3] = seg_data_post[:, :, ax_slice]
        image_tensor[1, ind+num_slices, 1] = seg_data_pre[:, cor_slice, :]
        image_tensor[1, ind+num_slices, 3] = seg_data_post[:, cor_slice, :]

    # Imshow arguments
    imshow_args = [layer_1_args, layer_2_args]

    grid_plot(
            image_tensor=image_tensor,
            imshow_args=imshow_args,
            header=header,
            col_titles=col_titles,
            row_titles=row_titles,
            outfile=outfile,
            legend_handles=patches
            )


def plot_pipeline(patient_identifiers: List[str], exam_dirs_preop: List[Path], exam_dirs_followup: List[Path], outfile: str,
                  classes_of_interest: List[int] = [1, 2, 3]) -> None:

    n_layers = 5    # one layer for each imshow config
    modalities = ["t1c", "t1", "t2", "flair"]
    tissues = ["gm", "wm", "csf"]

    # Paths
    for ind, (patient_identifier, exam_dir_preop, exam_dir_followup) in enumerate(zip(patient_identifiers, exam_dirs_preop, exam_dirs_followup)):
        preop_converted_files = {modality: MODALITY_CONVERTED_SCHEMA.format(base_dir=exam_dir_preop, modality=modality) for modality in modalities}
        followup_converted_files = {modality: MODALITY_CONVERTED_SCHEMA.format(base_dir=exam_dir_followup, modality=modality) for modality in modalities}

        preop_stripped_files = {modality: MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir_preop, modality=modality) for modality in modalities}
        followup_stripped_files = {modality: MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir_followup, modality=modality) for modality in modalities}

        tumor_seg_file = TUMORSEG_SCHEMA.format(base_dir=exam_dir_preop)
        recurrence_seg_file = TUMORSEG_SCHEMA.format(base_dir=exam_dir_followup)

        tissue_seg_file = TISSUE_SEG_SCHEMA.format(base_dir=exam_dir_preop)
        tissue_pbmaps_files = {tissue: TISSUE_PBMAP_SCHEMA.format(base_dir=exam_dir_preop, tissue=tissue) for tissue in tissues}

        brain_mask_file = BRAIN_MASK_SCHEMA.format(base_dir=exam_dir_preop)
        tumor_mask_file = TUMORSEG_CORE_SCHEMA.format(base_dir=exam_dir_preop)

        longitudinal_t1c_file = LONGITUDINAL_WARP_SCHEMA.format(base_dir=exam_dir_followup)
        longitudinal_rec_file = RECURRENCE_SCHEMA.format(base_dir=exam_dir_followup)

        model_output_file = PREDICTION_OUTPUT_SCHEMA.format(base_dir=exam_dir_preop, algo_id="gliodil")

        standard_plan_file = STANDARD_PLAN_SCHEMA.format(base_dir=exam_dir_preop)
        model_plan_file = MODEL_PLAN_SCHEMA.format(base_dir=exam_dir_preop, algo_id="gliodil")

        # Load images
        t1c_data_pre = load_mri_data(preop_stripped_files["t1c"])
        seg_data_pre = load_mri_data(tumor_seg_file)
        seg_data_post = load_mri_data(recurrence_seg_file)
        longitudinal_rec = load_mri_data(longitudinal_rec_file)
        model_data = load_and_resample_mri_data(model_output_file, resample_params=t1c_data_pre.shape, interp_type=1)

        # Compute tumor center of mass
        center = compute_center_of_mass(seg_data_pre, t1c_data_pre, classes_of_interest)
        ax_slice = center[2]
        #t1c_converted_followup = load_and_resample_mri_data(followup_converted_files["t1c"], resample_params=t1c_data_pre.shape, interp_type=1)[:, :, ax_slice]

        if ind == 0:
            # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
            cmap, norm, patches = get_cmap_norm_patches_tumorseg(classes_of_interest)

            # Titles
            row_titles = ["", "", ""]
            col_titles = ["T1c + Tumor", "Flair + Tumor", "T1c + Recurrence", "Concentration Prediction", "Model Plan", "Standard Plan"] + patient_identifiers[1:4]
            header = (
                    ""
                    )

            # Build image tensor
            image_tensor = np.empty((n_layers, 3, 3), dtype=object)

            # Layer 1: T1c, Flair
            layer_1_args = {"cmap": "gray", "interpolation": "none"}
            image_tensor[0, 0, 0] = load_mri_data(preop_stripped_files["t1c"])[:, :, ax_slice]
            image_tensor[0, 0, 1] = load_mri_data(preop_stripped_files["flair"])[:, :, ax_slice]
            image_tensor[0, 0, 2] = load_mri_data(longitudinal_t1c_file)[:, :, ax_slice]

            image_tensor[0, 1, 0] = load_mri_data(preop_stripped_files["t1c"])[:, :, ax_slice]
            image_tensor[0, 1, 1] = load_mri_data(longitudinal_t1c_file)[:, :, ax_slice]
            image_tensor[0, 1, 2] = load_mri_data(longitudinal_t1c_file)[:, :, ax_slice]

            # Layer 2: Tumor segmentations
            layer_2_args = {"cmap": cmap, "norm": norm, "alpha": 0.7, "interpolation": "none"}
            image_tensor[1, 0, 0] = load_mri_data(tumor_seg_file)[:, :, ax_slice]
            image_tensor[1, 0, 1] = load_mri_data(tumor_seg_file)[:, :, ax_slice]
            image_tensor[1, 0, 2] = load_mri_data(longitudinal_rec_file)[:, :, ax_slice]

            image_tensor[1, 1, 1] = load_mri_data(longitudinal_rec_file)[:, :, ax_slice]
            image_tensor[1, 1, 2] = load_mri_data(longitudinal_rec_file)[:, :, ax_slice]

            # Layer 3: Model pred
            layer_3_args = {"cmap": "inferno", "alpha": 0.80, "vmin": 0.0, "vmax": 1.0, "interpolation": "none"}
            image_tensor[2, 1, 0] = model_data[:, :, ax_slice]

            # Layer 4: Standard Plans
            layer_4_args = {"cmap": mcolors.ListedColormap([(0, 0, 0, 0), (34/255., 139/255., 34/255., 1)]), "alpha": 0.3, "vmin": 0.0, "vmax": 1.0}
            image_tensor[3, 1, 1] = load_mri_data(standard_plan_file)[:, :, ax_slice]

            # Layer 5: Model Plans
            layer_5_args = {"cmap": mcolors.ListedColormap([(0, 0, 0, 0), (210/255., 43/255., 43/255., 1)]), "alpha": 0.3, "vmin": 0.0, "vmax": 1.0}
            image_tensor[4, 1, 2] = load_mri_data(model_plan_file)[:, :, ax_slice]

        else:
            x_pos = ind - 1
            image_tensor[0, 2, x_pos] = load_mri_data(longitudinal_t1c_file)[:, :, ax_slice]
            image_tensor[1, 2, x_pos] = load_mri_data(longitudinal_rec_file)[:, :, ax_slice]
            image_tensor[3, 2, x_pos] = load_mri_data(standard_plan_file)[:, :, ax_slice]
            image_tensor[4, 2, x_pos] = load_mri_data(model_plan_file)[:, :, ax_slice]

    # Imshow arguments
    imshow_args = [layer_1_args, layer_2_args, layer_3_args, layer_4_args, layer_5_args]
    cmap, norm, patches = get_cmap_norm_patches_tumorseg_5(classes_of_interest)

    grid_plot(
            image_tensor=image_tensor,
            imshow_args=imshow_args,
            header=header,
            col_titles=col_titles,
            row_titles=row_titles,
            outfile=outfile,
            legend_handles=patches
            )


def plot_plans(patient_identifier: str, exam_identifier_pre: str, exam_identifier_followup: str,
               exam_dir_preop: Path, exam_dir_followup: Path, outfile: str,
               classes_of_interest: List[int] = [1, 2, 3]) -> None:

    n_layers = 3    # one layer for each imshow config

    # Paths
    longitudinal_t1c_file = LONGITUDINAL_WARP_SCHEMA.format(base_dir=exam_dir_followup)
    longitudinal_rec_file = RECURRENCE_SCHEMA.format(base_dir=exam_dir_followup)
    model_output_file = PREDICTION_OUTPUT_SCHEMA.format(base_dir=exam_dir_preop, algo_id="sbtc")
    standard_plan_file = STANDARD_PLAN_SCHEMA.format(base_dir=exam_dir_preop)
    model_plan_file = MODEL_PLAN_SCHEMA.format(base_dir=exam_dir_preop, algo_id="sbtc")

    # Load images
    t1c_data_post = load_mri_data(longitudinal_t1c_file)
    longitudinal_rec = load_mri_data(longitudinal_rec_file)
    model_data = load_and_resample_mri_data(model_output_file, resample_params=t1c_data_post.shape, interp_type=1)
    standard_plan = load_mri_data(standard_plan_file)
    model_plan = load_mri_data(model_plan_file)

    # Ignore resection cavity label
    longitudinal_rec[longitudinal_rec==4] = 0

    # Compute tumor center of mass
    center = compute_center_of_mass(longitudinal_rec, t1c_data_post, classes_of_interest)
    step_size = 10
    num_slices = 5
    patient_dim = t1c_data_post.shape
    axial_slices, coronal_slices = get_slices(center, num_slices, step_size, patient_dim)

    # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
    cmap, norm, patches = get_cmap_norm_patches_tumorseg(classes_of_interest)

    # Titles
    col_titles = ["T1c", "T1", "T2", "Flair"]
    row_titles = axial_slices + coronal_slices
    header = (
            f"Patient: {patient_identifier}\n"
            f"Exam (preop): {exam_identifier_pre}\n"
            f"Exam (postop): {exam_identifier_followup}\n"
            f"CoM slice (axial/coronal): {center[2]}/{center[1]}\n"
            )

    # Build image tensor
    image_tensor = np.empty((n_layers, num_slices*2, 4), dtype=object)

    # Layer 1: T1c, T1c, T1c, Tissueseg
    layer_1_args = {"cmap": "gray", "interpolation": "none"}

    layer_1_args = {"cmap": "gray"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        image_tensor[0, ind, 0] = t1c_data_post[:, :, ax_slice]
        image_tensor[0, ind, 1] = t1c_data_post[:, :, ax_slice]
        image_tensor[0, ind, 2] = t1c_data_post[:, :, ax_slice]
        image_tensor[0, ind, 3] = t1c_data_post[:, :, ax_slice]

        image_tensor[0, ind+num_slices, 0] = t1c_data_post[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 1] = t1c_data_post[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 2] = t1c_data_post[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 3] = t1c_data_post[:, cor_slice, :]

    layer_2_args = {"cmap": cmap, "norm": norm, "alpha": 0.9, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        image_tensor[1, ind, 0] = longitudinal_rec[:, :, ax_slice]
        image_tensor[1, ind, 1] = standard_plan[:, :, ax_slice]
        image_tensor[1, ind, 2] = model_plan[:, :, ax_slice]

        image_tensor[1, ind+num_slices, 0] = longitudinal_rec[:, cor_slice, :]
        image_tensor[1, ind+num_slices, 1] = standard_plan[:, cor_slice, :]
        image_tensor[1, ind+num_slices, 2] = model_plan[:, cor_slice, :]

    layer_3_args = {"cmap": "inferno", "alpha": 0.90, "vmin": 0.0, "vmax": 1.0, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        image_tensor[2, ind, 3] = model_data[:, :, ax_slice]
        image_tensor[2, ind+num_slices, 3] = model_data[:, cor_slice, :]

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


def plot_full_brain(patient_identifier: str, exam_identifier_pre: str, exam_identifier_followup: str,
                    exam_dir_preop: Path, exam_dir_followup: Path, outfile: str,
                    classes_of_interest: List[int] = [1, 2, 3]) -> None:

    n_layers = 3    # one layer for each imshow config
    modalities = ["t1c", "t1", "t2", "flair"]

    # Paths
    preop_stripped_files = {modality: MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir_preop, modality=modality) for modality in modalities}
    followup_stripped_files = {modality: MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir_followup, modality=modality) for modality in modalities}
    tumor_seg_file = TUMORSEG_SCHEMA.format(base_dir=exam_dir_preop)
    recurrence_seg_file = TUMORSEG_SCHEMA.format(base_dir=exam_dir_followup)
    standard_plan_file = STANDARD_PLAN_SCHEMA.format(base_dir=exam_dir_preop)

    # Load images
    t1c_pre = load_mri_data(preop_stripped_files["t1c"])
    t1c_post = load_mri_data(followup_stripped_files["t1c"])
    
    try:
        flair_pre = load_mri_data(preop_stripped_files["flair"])
    except:
        flair_pre = np.ones(t1c_pre.shape)
        print(f"Preop FLAIR MRI not found. Conitnuing with empty image.")
    try:
        flair_post = load_mri_data(followup_stripped_files["flair"])
    except:
        flair_post = np.ones(t1c_post.shape)
        print(f"Followup FLAIR MRI not found. Conitnuing with empty image.")

    tumor_seg = load_mri_data(tumor_seg_file)
    recurrence_seg = load_mri_data(recurrence_seg_file)
    
    try:
        standard_plan = load_mri_data(standard_plan_file)
    except:
        standard_plan = np.zeros(t1c_pre.shape)
        print(f"Standard plan not found. Continuing with emtpy image.")

    # Generate projections
    tumor_projections = [get_segmentation_projection(tumor_seg, label=label, axis=2) for label in classes_of_interest]
    recurrence_projections = [get_segmentation_projection(recurrence_seg, label=label, axis=2) for label in classes_of_interest]
    radplan_projection = get_segmentation_projection(standard_plan, label=1, axis=2)

    # Ignore resection cavity label
    recurrence_seg[recurrence_seg==4] = 0  # ignore cavity

    # Compute tumor center of mass
    #center = compute_center_of_mass(longitudinal_rec, t1c_data_post, classes_of_interest)
    center = [d // 2 for d in t1c_post.shape]
    step_size = 10
    num_slices = 15
    patient_dim = t1c_post.shape
    axial_slices = [k*10 for k in range(0, 15)]
    coronal_slices = axial_slices
    #axial_slices, coronal_slices = get_slices(center, num_slices, step_size, patient_dim)

    # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
    cmap, norm, patches = get_cmap_norm_patches_tumorseg(classes_of_interest)

    # Titles
    col_titles = ["Projection"] + axial_slices
    row_titles = ["T1c\n(preop)", "FLAIR\n(preop)", "TumorSeg\n(preop)", "RecurrenceSeg\n(followup)", "FLAIR\n(followup)", "T1c\n(followup)", "StandardPlan\n(followup)"]
    header = (
            f"Patient: {patient_identifier}\n"
            f"Exam (preop): {exam_identifier_pre}\n"
            f"Exam (postop): {exam_identifier_followup}\n"
            f"CoM slice (axial/coronal): {center[2]}/{center[1]}\n"
            )

    # Build image tensor
    image_tensor = np.empty((n_layers, len(row_titles), num_slices+1), dtype=object)

    # Layer 1: T1c, T1c, T1c, Tissueseg
    layer_1_args = {"cmap": "gray", "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        image_tensor[0, 0, ind+1] = t1c_pre[:, :, ax_slice]
        image_tensor[0, 1, ind+1] = flair_pre[:, :, ax_slice]
        image_tensor[0, 2, ind+1] = t1c_pre[:, :, ax_slice]
        image_tensor[0, 3, ind+1] = t1c_post[:, :, ax_slice]
        image_tensor[0, 4, ind+1] = flair_post[:, :, ax_slice]
        image_tensor[0, 5, ind+1] = t1c_post[:, :, ax_slice]
        image_tensor[0, 6, ind+1] = standard_plan[:, :, ax_slice]

    # Layer 2: Tumor segmentations
    layer_2_args = {"cmap": cmap, "norm": norm, "alpha": 0.9, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        image_tensor[1, 2, ind+1] = tumor_seg[:, :, ax_slice]
        image_tensor[1, 3, ind+1] = recurrence_seg[:, :, ax_slice]

    # Layer 3: Projections
    layer_3_args = {"cmap": "gray", "interpolation": "none"}
    image_tensor[2, 0, 0] = tumor_projections[0]
    image_tensor[2, 1, 0] = tumor_projections[1]
    image_tensor[2, 2, 0] = tumor_projections[2]
    image_tensor[2, 3, 0] = recurrence_projections[0]
    image_tensor[2, 4, 0] = recurrence_projections[1]
    image_tensor[2, 5, 0] = recurrence_projections[2]
    image_tensor[2, 6, 0] = radplan_projection

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


def plot_difference(img1_file, img2_file, identifier, outfile) -> None:

    n_layers = 2

    # Load images
    img1 = load_mri_data(img1_file)
    img2 = load_mri_data(img2_file)
    diff = (img1 - img2)

    if img1.shape != img2.shape:
        raise ValueError(f"Dimension mismatch. Images need to be the same dimension.")

    center = [d // 2 for d in img1.shape]
    step_size = 10
    num_slices = 15
    axial_slices = [k*10 for k in range(0, 15)]
    coronal_slices = axial_slices

    # Titles
    col_titles = axial_slices
    row_titles = ["img1", "img2", "difference"]
    header = (
            f"Patient: {identifier}\n"
            f"Volume 1: {np.sum(img1 > 0)}\n"
            f"Volume 2: {np.sum(img2 > 0)}\n"
            f"Difference: {np.sum(diff > 0)}"
            )

    # Build image tensor
    image_tensor = np.empty((n_layers, len(row_titles), len(col_titles)), dtype=object)

    # Layer 1: T1c, T1c, T1c, Tissueseg
    layer_1_args = {"cmap": "gray", "interpolation": "none", "vmin": 0, "vmax": 1}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        image_tensor[0, 0, ind] = img1[:, :, ax_slice]
        image_tensor[0, 1, ind] = img2[:, :, ax_slice]

    layer_2_args = {"cmap": "inferno", "vmin": -1.0, "vmax": 1.0, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        image_tensor[1, 2, ind] = diff[:, :, ax_slice]

    # Imshow arguments
    imshow_args = [layer_1_args, layer_2_args]

    grid_plot(
            image_tensor=image_tensor,
            imshow_args=imshow_args,
            header=header,
            col_titles=col_titles,
            row_titles=row_titles,
            outfile=outfile
            )


def plot_tumor_sizes(dataset_ids, dataset_dirs, dataset_rootdirs, outfile, recurrence=False):
    xvals = []
    tumor_sizes = []
    jitter = 0.2

    ind = 1
    for d_id, d_d, d_rd in zip(dataset_ids, dataset_dirs, dataset_rootdirs):
        dataset = LongitudinalDataset(dataset_id=d_id, root_dir=d_rd)
        dataset.load(d_d)

        logger.info(f"Processing {d_id}...")

        no_recurrence_counter = 0
        for patient in dataset.patients:
            patient_id = patient["patient_id"]
            preop_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]  # Find first preop exam
            followup_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="followup")[0]

            if "UPENN" in d_id:
                preop_exam_dir = preop_exam["t1"].parent
                followup_exam_dir = followup_exam["t1"].parent
            elif "GLIODIL" in d_id:
                if patient_id in NO_T1C:
                    preop_exam_dir = preop_exam["tumorseg"].parent / "preop"
                    followup_exam_dir = followup_exam["tumorseg"].parent / "followup"
                else:
                    preop_exam_dir = preop_exam["t1c"].parent / "preop"
                    followup_exam_dir = followup_exam["t1c"].parent / "followup"
            else:
                preop_exam_dir = preop_exam["t1c"].parent
                followup_exam_dir = followup_exam["t1c"].parent

            try:
                if recurrence:
                    tumorseg_dir = RECURRENCE_SCHEMA.format(base_dir=followup_exam_dir)
                else:
                    tumorseg_dir = TUMORSEG_SCHEMA.format(base_dir=preop_exam_dir)
                tumorseg = np.rint(nib.load(tumorseg_dir).get_fdata()).astype(np.int32)
                tumorcore = ((tumorseg==1) | (tumorseg==3)).astype(np.int32)
                tumor_size = np.sum(tumorcore) / 1000

                if tumor_size == 0:
                    no_recurrence_counter += 1

                xvals.append(ind)
                tumor_sizes.append(tumor_size)
            except:
                print(f"File not found: {tumorseg_dir}")
                continue
        ind += 1
        print(f"{no_recurrence_counter} empty recurrence segmentations for {d_id}.")

    logger.info(f"Generating plot...")

    print(f"Tumor size (mean): {np.mean(tumor_sizes)} \u00B1 {stats.stdev(tumor_sizes)}")
    print(f"Tumor size (median): {stats.median(tumor_sizes)}")

    dataset_ids.append("PREDICT-GBM")
    tumor_sizes_new = tumor_sizes.copy()
    for ts in tumor_sizes:
        tumor_sizes_new.append(ts)
        xvals.append(len(dataset_ids))
    tumor_sizes = tumor_sizes_new

    boxplot_input = [[] for ind in dataset_ids]
    for d_ind, tsize in zip(xvals, tumor_sizes):
        boxplot_input[d_ind-1].append(tsize)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(
            boxplot_input,
            positions=list(range(1, len(dataset_ids)+1)),
            widths=0.6,
            showfliers=False,
            showmeans=False,
            sym="",
            medianprops={"color": (1, 127/255, 0, 1)}
            )
    xvals_scatter = [xv+random.uniform(-1*jitter, jitter) for xv in xvals]
    ax.scatter(xvals_scatter, tumor_sizes, alpha=0.5, s=50, edgecolors="none", linewidths=0)
    ax.set_xticks(range(len(dataset_ids)+1))
    dataset_ids = [""] + [d_id.replace("GLIODIL", "TUM-GBM") for d_id in dataset_ids]
    ax.set_xticklabels(dataset_ids, rotation=25, ha="right")
    ax.set_ylabel(f"Tumor size [cm$^{{3}}$] ({'recurrence' if recurrence else 'preop'})")
    fig.tight_layout()

    fig.savefig(outfile)


def plot_com_distances(dataset_ids, dataset_dirs, dataset_rootdirs, outfile):
    xvals = []
    com_distances = []
    jitter = 0.2

    ind = 1
    for d_id, d_d, d_rd in zip(dataset_ids, dataset_dirs, dataset_rootdirs):
        dataset = LongitudinalDataset(dataset_id=d_id, root_dir=d_rd)
        dataset.load(d_d)

        logger.info(f"Processing {d_id}...")

        for patient in dataset.patients:
            patient_id = patient["patient_id"]
            preop_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]  # Find first preop exam
            followup_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="followup")[0]

            if "UPENN" in d_id:
                preop_exam_dir = preop_exam["t1"].parent
                followup_exam_dir = followup_exam["t1"].parent
            elif "GLIODIL" in d_id:
                if patient_id in NO_T1C:
                    preop_exam_dir = preop_exam["tumorseg"].parent / "preop"
                    followup_exam_dir = followup_exam["tumorseg"].parent / "followup"
                else:
                    preop_exam_dir = preop_exam["t1c"].parent / "preop"
                    followup_exam_dir = followup_exam["t1c"].parent / "followup"
            else:
                preop_exam_dir = preop_exam["t1c"].parent
                followup_exam_dir = followup_exam["t1c"].parent

            try:
                tumorseg_dir = TUMORSEG_SCHEMA.format(base_dir=preop_exam_dir)
                recurrence_dir = RECURRENCE_SCHEMA.format(base_dir=followup_exam_dir)
                
                tumorseg = np.rint(nib.load(tumorseg_dir).get_fdata()).astype(np.int32)
                recurrence = np.rint(nib.load(recurrence_dir).get_fdata()).astype(np.int32)

                com_tumor = compute_center_of_mass(tumorseg, tumorseg, classes=[1,2,3])
                com_recurrence = compute_center_of_mass(recurrence, recurrence, classes=[1,2,3])
                distance = math.dist(com_tumor, com_recurrence) / 10

                xvals.append(ind)
                com_distances.append(distance)
            except Exception as e:
                raise e
        ind += 1

    logger.info(f"Generating plot...")
    
    print(f"Distance (mean): {np.mean(com_distances)} \u00B1 {stats.stdev(com_distances)}")
    print(f"Distance (median): {stats.median(com_distances)}")

    dataset_ids.append("PREDICT-GBM")
    com_distances_new = com_distances.copy()
    for cdis in com_distances:
        com_distances_new.append(cdis)
        xvals.append(len(dataset_ids))
    com_distances = com_distances_new

    boxplot_input = [[] for ind in dataset_ids]
    for d_ind, dist in zip(xvals, com_distances):
        boxplot_input[d_ind-1].append(dist)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(
            boxplot_input,
            positions=list(range(1, len(dataset_ids)+1)),
            widths=0.6,
            showfliers=False,
            showmeans=False,
            sym="",
            medianprops={"color": (1, 127/255, 0, 1)}
            )
    xvals_scatter = [xv+random.uniform(-1*jitter, jitter) for xv in xvals]
    ax.scatter(xvals_scatter, com_distances, alpha=0.5, s=50, edgecolors="none", linewidths=0)
    ax.axhline(y=1.5, color="0.4", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(dataset_ids)+1))
    dataset_ids = [""] + [d_id.replace("GLIODIL", "TUM-GBM") for d_id in dataset_ids]
    ax.set_xticklabels(dataset_ids, rotation=25, ha="right")
    ax.set_ylabel(f"Distance tum.-rec. [cm]")
    fig.tight_layout()

    fig.savefig(outfile)


def plot_performances(dataset_ids, dataset_dirs, dataset_rootdirs, model_id, outfile):
    performances_by_dataset = {d_id: [] for d_id in dataset_ids}
    performances_by_dataset["PREDICT-GBM"] = []

    for d_id, d_d, d_rd in zip(dataset_ids, dataset_dirs, dataset_rootdirs):
        dataset = LongitudinalDataset(dataset_id=d_id, root_dir=d_rd)
        dataset.load(d_d)

        logger.info(f"Processing {d_id}...")

        for patient in dataset.patients:
            patient_id = patient["patient_id"]
            preop_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]  # Find first preop exam
            followup_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="followup")[0]

            if "UPENN" in d_id:
                preop_exam_dir = preop_exam["t1"].parent
                followup_exam_dir = followup_exam["t1"].parent
            elif "GLIODIL" in d_id:
                if patient_id in NO_T1C:
                    preop_exam_dir = preop_exam["tumorseg"].parent / "preop"
                    followup_exam_dir = followup_exam["tumorseg"].parent / "followup"
                else:
                    preop_exam_dir = preop_exam["t1c"].parent / "preop"
                    followup_exam_dir = followup_exam["t1c"].parent / "followup"
            else:
                preop_exam_dir = preop_exam["t1c"].parent
                followup_exam_dir = followup_exam["t1c"].parent

            try:
                performance_dir = METRICS_SCHEMA.format(base_dir=followup_exam_dir, algo_id=model_id.lower())
                performance_dict = json.load(open(performance_dir, "r"))
                performances_by_dataset[d_id].append((
                    performance_dict["recurrence_coverage_standard"]*100,
                    performance_dict["recurrence_coverage_model"]*100,
                    performance_dict["recurrence_coverage_standard_all"]*100,
                    performance_dict["recurrence_coverage_model_all"]*100
                    ))
                performances_by_dataset["PREDICT-GBM"].append((
                    performance_dict["recurrence_coverage_standard"]*100,
                    performance_dict["recurrence_coverage_model"]*100,
                    performance_dict["recurrence_coverage_standard_all"]*100,
                    performance_dict["recurrence_coverage_model_all"]*100
                    ))

            except Exception as e:
                #raise e
                print(f"Excpetion for {followup_exam_dir}: {e}")

    logger.info(f"Generating plot...")

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), sharex=True, sharey=True)

    dataset_ids.append("PREDICT-GBM")
    for ind, (ax, d_id) in enumerate(zip(axes.flat, dataset_ids)):
        xs, ys, xs_all, ys_all = zip(*performances_by_dataset[d_id])
        ax.scatter(xs, ys, alpha=0.6, color=(238/255.0, 165/255.0, 62/255.0), label="Enhancing",  s=50, edgecolors="none", linewidths=0)
        #ax.scatter(xs_all, ys_all, alpha=0.5, color=(57/255.0, 118/255.0, 129/255.0), label="Any", s=50, edgecolors="none", linewidths=0)
        ax.scatter(xs_all, ys_all, alpha=0.6, label="Any", s=50, edgecolors="none", linewidths=0)
        ax.plot([0,100], [0,100], linewidth=1, linestyle="--", color="black")
        d_id = d_id.replace("GLIODIL", "TUM-GBM")
        ax.set_title(d_id, fontweight="bold")

        if ind % 3 == 0:
            ax.set_ylabel(f"Recurrence coverage ({model_id}) [%]")
        if ind>5:
            ax.set_xlabel("Recurrence coverage (standard) [%]")

    ax.legend(loc="lower right")
    fig.delaxes(axes[2][2])
    fig.tight_layout()

    # uncomment to center bottom images
    #fig.set_constrained_layout(False)
    #x0, [cm$^{{3}}$]y0, w, h = axes[2,0].get_position().bounds
    #dx = 0.15
    #axes[2,0].set_position([x0 + dx, y0, w, h])

    #x0, y0, w, h = axes[2,1].get_position().bounds
    #dx = 0.15
    #axes[2,1].set_position([x0 + dx, y0, w, h])

    fig.savefig(outfile)


def plot_missed(dataset_ids, dataset_dirs, dataset_rootdirs, model_id, outfile):
    performances_by_dataset = {d_id: [] for d_id in dataset_ids}
    performances_by_dataset["PREDICT-GBM"] = []

    for d_id, d_d, d_rd in zip(dataset_ids, dataset_dirs, dataset_rootdirs):
        dataset = LongitudinalDataset(dataset_id=d_id, root_dir=d_rd)
        dataset.load(d_d)

        logger.info(f"Processing {d_id}...")

        for patient in dataset.patients:
            patient_id = patient["patient_id"]
            preop_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]  # Find first preop exam
            followup_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="followup")[0]

            if "UPENN" in d_id:
                preop_exam_dir = preop_exam["t1"].parent
                followup_exam_dir = followup_exam["t1"].parent
            elif "GLIODIL" in d_id:
                if patient_id in NO_T1C:
                    preop_exam_dir = preop_exam["tumorseg"].parent / "preop"
                    followup_exam_dir = followup_exam["tumorseg"].parent / "followup"
                else:
                    preop_exam_dir = preop_exam["t1c"].parent / "preop"
                    followup_exam_dir = followup_exam["t1c"].parent / "followup"
            else:
                preop_exam_dir = preop_exam["t1c"].parent
                followup_exam_dir = followup_exam["t1c"].parent

            try:
                performance_dir = METRICS_SCHEMA.format(base_dir=followup_exam_dir, algo_id=model_id.lower())
                performance_dict = json.load(open(performance_dir, "r"))
                performances_by_dataset[d_id].append((
                    performance_dict["missed_voxels_standard"],
                    performance_dict["missed_voxels_model"],
                    performance_dict["missed_voxels_standard_all"],
                    performance_dict["missed_voxels_model_all"]
                    ))
                performances_by_dataset["PREDICT-GBM"].append((
                    performance_dict["missed_voxels_standard"],
                    performance_dict["missed_voxels_model"],
                    performance_dict["missed_voxels_standard_all"],
                    performance_dict["missed_voxels_model_all"]
                    ))

            except Exception as e:
                #raise e
                print(f"Excpetion for {followup_exam_dir}: {e}")

    logger.info(f"Generating plot...")

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), sharex=True, sharey=True)

    dataset_ids.append("PREDICT-GBM")
    for ind, (ax, d_id) in enumerate(zip(axes.flat, dataset_ids)):
        xs, ys, xs_all, ys_all = zip(*performances_by_dataset[d_id])
        ax.scatter(xs, ys, alpha=0.6, color=(238/255.0, 165/255.0, 62/255.0), label="Enhancing",  s=50, edgecolors="none", linewidths=0)
        #ax.scatter(xs_all, ys_all, alpha=0.6, label="Any", s=50, edgecolors="none", linewidths=0)
        ax.plot([0, 10**5], [0, 10**5], linewidth=1, linestyle="--", color="black")
        ax.set_title(d_id, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")

        if ind % 3 == 0:
            ax.set_ylabel(f"Missed volume ({model_id}) [mm$^{{3}}$]")
        if ind>5:
            ax.set_xlabel("Missed volume (standard) [mm$^{{3}}$]")

    ax.legend()
    fig.delaxes(axes[2][2])
    fig.tight_layout()

    # uncomment to center bottom images
    #fig.set_constrained_layout(False)
    #x0, y0, w, h = axes[2,0].get_position().bounds
    #dx = 0.15
    #axes[2,0].set_position([x0 + dx, y0, w, h])

    #x0, y0, w, h = axes[2,1].get_position().bounds
    #dx = 0.15
    #axes[2,1].set_position([x0 + dx, y0, w, h])

    fig.savefig(outfile)


def plot_diff_vs_distance(dataset_ids, dataset_dirs, dataset_rootdirs, model_id, outfile):
    differences = []
    distances = []

    for d_id, d_d, d_rd in zip(dataset_ids, dataset_dirs, dataset_rootdirs):
        dataset = LongitudinalDataset(dataset_id=d_id, root_dir=d_rd)
        dataset.load(d_d)

        logger.info(f"Processing {d_id}...")

        for patient in dataset.patients:
            patient_id = patient["patient_id"]
            preop_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]  # Find first preop exam
            followup_exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="followup")[0]

            if "UPENN" in d_id:
                preop_exam_dir = preop_exam["t1"].parent
                followup_exam_dir = followup_exam["t1"].parent
            elif "GLIODIL" in d_id:
                if patient_id in NO_T1C:
                    preop_exam_dir = preop_exam["tumorseg"].parent / "preop"
                    followup_exam_dir = followup_exam["tumorseg"].parent / "followup"
                else:
                    preop_exam_dir = preop_exam["t1c"].parent / "preop"
                    followup_exam_dir = followup_exam["t1c"].parent / "followup"
            else:
                preop_exam_dir = preop_exam["t1c"].parent
                followup_exam_dir = followup_exam["t1c"].parent

            try:
                tumorseg_dir = TUMORSEG_SCHEMA.format(base_dir=preop_exam_dir)
                recurrence_dir = RECURRENCE_SCHEMA.format(base_dir=followup_exam_dir)
                tumorseg = np.rint(nib.load(tumorseg_dir).get_fdata()).astype(np.int32)
                recurrence = np.rint(nib.load(recurrence_dir).get_fdata()).astype(np.int32)

                com_tumor = compute_center_of_mass(tumorseg, tumorseg, classes=[1,2,3])
                com_recurrence = compute_center_of_mass(recurrence, recurrence, classes=[1,2,3])
                distance = math.dist(com_tumor, com_recurrence) / 10

                performance_dir = METRICS_SCHEMA.format(base_dir=followup_exam_dir, algo_id=model_id.lower())
                performance_dict = json.load(open(performance_dir, "r"))
                difference = (performance_dict["recurrence_coverage_model"] - performance_dict["recurrence_coverage_standard"]) * 100
                
                differences.append(difference)
                distances.append(distance)

            except Exception as e:
                #raise e
                print(f"Excpetion for {followup_exam_dir}: {e}")

    logger.info(f"Generating plot...")
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(distances, differences, alpha=0.5, s=50, edgecolors="none", linewidths=0)
    #ax.axhline(y=1.5, color="0.4", linestyle="--", linewidth=1)
    ax.set_ylabel(r"Coverage$_{std}$ - Coverage$_{model}$")
    ax.set_xlabel(r"Distance tum.-rec. [cm]")
    fig.tight_layout()

    fig.savefig(outfile)


if __name__ == "__main__":
    # Example:
    # python gbm_bench/utils/visualization.py

    """
    patient_identifiers = ["RHUH-025", "LUMIERE-008", "RHUH-018", "UPENN-140"]
    exam_dirs_preop = [
            Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0025/10-14-2012-NA-RM DE CEREBRO SINCON CONTRASTE-82954"),
            Path("/mnt/Drive2/lucas/datasets/LUMIERE/Imaging/Patient-008/week-000"),
            Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0018/06-25-2017-NA-RM CEREBRAL-66931/"),
            Path("/mnt/Drive2/lucas/datasets/UPENN-GBM/UPENN-GBM/UPENN-GBM-00140/03-15-2009-NA-BrainTumor-11707")
            ]
    exam_dirs_followup = [
            Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0025/03-24-2013-NA-CEREBRAL-48696"),
            Path("/mnt/Drive2/lucas/datasets/LUMIERE/Imaging/Patient-008/week-017"),
            Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0018/11-15-2017-NA-CRANEO-18771"),
            Path("/mnt/Drive2/lucas/datasets/UPENN-GBM/UPENN-GBM/UPENN-GBM-00140/05-09-2010-NA-BrainTumor-75173")
            ]
    """

    patient_identifiers = ["TUM-GBM-097", "RHUH-005", "RHUH-011", "LUMIERE-066"]
    exam_dirs_preop = [
            Path("/mnt/Drive2/lucas/datasets/GLIODIL/respond_tum_097/d0/preop"),
            Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0005/05-25-2013-NA-RM CRANEONEURONAVEGADOR-29811/"),
            Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0011/08-20-2017-NA-RM CRANEO-91051/"),
            Path("/mnt/Drive2/lucas/datasets/LUMIERE/Imaging/Patient-066/week-000")
            ]
    exam_dirs_followup = [
            Path("/mnt/Drive2/lucas/datasets/GLIODIL/respond_tum_097/d205/followup"),
            Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0005/09-14-2014-NA-CEREBRAL-38655"),
            Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0011/11-01-2017-NA-RM CEREBRAL-80880/"),
            Path("/mnt/Drive2/lucas/datasets/LUMIERE/Imaging/Patient-066/week-101")
            ]
    """
    plot_model_multislice(
            patient_identifier="RHUH-0001",
            exam_identifier="01-25-2015",
            algorithm_identifier="LMI",
            exam_dir=Path("test_data/exam1/"),
            outfile="tmp_visualization/test_multislice.pdf"
            )

    plot_recurrence_multislice(
            patient_identifier="RHUH-0001",
            exam_identifier_pre="Pre",
            exam_identifier_followup="Post",
            exam_dir_preop=Path("test_data/exam1/"),
            exam_dir_followup=Path("test_data/exam3/"),
            outfile="tmp_visualization/test_longitudinal.pdf"
            )
    """

    plot_pipeline(
            patient_identifiers=patient_identifiers,
            exam_dirs_preop=exam_dirs_preop,
            exam_dirs_followup=exam_dirs_followup,
            outfile="tmp_visualization/pipeline.pdf"
            )
    
    """
    plot_plans(
            patient_identifier="RHUH-0011",
            exam_identifier_pre="Pre",
            exam_identifier_followup="Post",
            exam_dir_preop=Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0008/09-27-2015-NA-Craneo-26679"),
            exam_dir_followup=Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0008/04-05-2016-NA-RM CEREBRO-94961"),
            outfile="tmp_visualization/plans.pdf"
            )
    
    plot_full_brain(
            patient_identifier="RHUH-0024",
            exam_identifier_pre="Pre",
            exam_identifier_followup="Post",
            exam_dir_preop=Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0024/11-10-2013-NA-Craneo-58463"),
            exam_dir_followup=Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0024/01-29-2014-NA-RM CEREBRO-96283"),
            outfile="tmp_visualization/qualitycontrol.pdf"
            )
    
    plot_difference(
            img1_file="/home/home/lucas/jonasplans/standardPlan.nii.gz",
            img2_file="/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0012/0/processed/tumor_segmentation/standard_plan.nii.gz",
            identifier="tgm016",,
            outfile="tmp/standard_difference.pdf")
    """
