from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import nibabel as nib
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"


def visualize_data(files: List[Path], label: str, titles: Optional[List[str]] = None):
    """Visualize the MRI modalities

    Args:
        files (List[Path]): List of paths to the MRI modalities
        label (str): Label for the y-axis on the far left left, i.e. category of the passed images (e.g. input, output)
        titles (Optional[List[str]], optional): Title of images. Defaults to None.
    """
    fig, axes = plt.subplots(1, len(files), figsize=(len(files) * 3, 7))

    for i, file in enumerate(files):
        modality_np = nib.load(file).get_fdata().transpose(2, 1, 0)
        axes[i].set_title(titles[i] if titles else file.name)
        axes[i].imshow(modality_np[modality_np.shape[0] // 2, :, :], cmap="gray")
    axes[0].set_ylabel(label)
    fig.savefig(f"preprocessed/{label}.png".replace(" ", "_"))


def visualize_defacing(
    file: Path,
):
    """Visualize the defacing of the MRI modality

    Args:
        file (Path): Path to the MRI modality
    """

    modality_np = nib.load(file).get_fdata().transpose(2, 1, 0)
    plt.figure(figsize=(4, 5))
    plt.title(file.name)
    plt.imshow(modality_np[:, ::-1, 75], cmap="gray", origin="lower")


# specify input and output paths
data_folder = Path("/home/home/lucas/data/RHUH-GBM/Images/NIfTI/RHUH-GBM_nii_v1/RHUH-0001/0")
conversion_folder = Path("/home/home/lucas/testing/nifti_conv")
registration_folder = Path("/home/home/lucas/testing/preprocessed/atlas_registration")
output_dir = Path("preprocessed")

t1c_file = data_folder / "RHUH-0001_0_t1ce.nii.gz"
t1_file = data_folder / "RHUH-0001_0_t1.nii.gz"
fla_file = data_folder / "RHUH-0001_0_flair.nii.gz"
t2_file = data_folder / "RHUH-0001_0_t2.nii.gz"

t1c_conv_file = conversion_folder / "t1c" / "Ax_T1_3d_NEURONAVEGADOR_12.000000-Ax_T1_3d_NEURONAVEGADOR-55128_RHUH-0001_GE_RHUH-0001_DTI+NEURONAVEGADOR_3_GR_12_.nii.gz"
t1_conv_file = conversion_folder / "t1" / "Ax_T1_FSE_5.000000-Ax_T1_FSE-08383_RHUH-0001_GE_RHUH-0001_Craneo_Rutina_7_SE_5_.nii.gz"
fla_conv_file = conversion_folder / "flair" / "Ax_T2_FLAIR_3.000000-Ax_T2_FLAIR-62646_RHUH-0001_GE_RHUH-0001_Craneo_Rutina_4_SE_IR_3_.nii.gz"
t2_conv_file = conversion_folder / "t2" / "Ax_T2_FRFSE_6.000000-Ax_T2_FRFSE-46501_RHUH-0001_GE_RHUH-0001_Craneo_Rutina_6_SE_6_.nii.gz"

t1c_reg_file = registration_folder / "atlas__t1c.nii.gz"
t1_reg_file = registration_folder / "atlas__t1.nii.gz"
fla_reg_file = registration_folder / "atlas__flair.nii.gz"
t2_reg_file = registration_folder / "atlas__t2.nii.gz"

t1c_normalized_skull_output_path = output_dir / "norm_skull_dir" / "t1c_skull_normalized.nii.gz"
t1c_normalized_bet_output_path = output_dir / "norm_bet_dir" / "t1c_bet_normalized.nii.gz"
#t1c_normalized_defaced_output_path = output_dir / "t1c_normalized_defaced.nii.gz"
t1c_bet_mask = output_dir / "masks" / "t1c_bet_mask.nii.gz"
#t1c_defacing_mask = output_dir / "t1c_defacing_mask.nii.gz"

t1_normalized_bet_output_path = output_dir / "norm_bet_dir" / "t1_bet_normalized.nii.gz"
fla_normalized_bet_output_path = output_dir / "norm_bet_dir" / "flair_bet_normalized.nii.gz"
t2_normalized_bet_output_path = output_dir / "norm_bet_dir" / "t2_bet_normalized.nii.gz"


# Dataset Nifti
visualize_data(
        files=[
            t1c_file,
            t1_file,
            fla_file,
            t2_file],
        titles=[
            "T1c",
            "T1",
            "Flair",
            "T2"
            ],
        label="Dataset Nifti")

# Dataset DICOM converted
visualize_data(
        files=[
            t1c_conv_file,
            t1_conv_file,
            fla_conv_file,
            t2_conv_file],
        titles=[
            "T1c",
            "T1",
            "Flair",
            "T2"
            ],
        label="Converted")

# Data after registration
visualize_data(
        files=[
            t1c_reg_file,
            t1_reg_file,
            fla_reg_file,
            t2_reg_file],
        titles=[
            "T1c",
            "T1",
            "Flair",
            "T2"
            ],
        label="Post Registration")

# inspect the different outputs for the center modality (normalized atlas registered with skull, brain extracted (bet) and defaced)
visualize_data(
    files=[
        t1c_normalized_skull_output_path,
        t1c_normalized_bet_output_path,
        #t1c_normalized_defaced_output_path,
    ],
    titles=[
        "T1c normalized",
        "T1c skull strippped"
        ],
    label="T1C outputs",
)

# inspect the different outputs for the moving modalities (normalized atlas registered brain extracted (bet))
visualize_data(
    files=[
        t1c_normalized_bet_output_path,
        t1_normalized_bet_output_path,
        fla_normalized_bet_output_path,
        t2_normalized_bet_output_path,
    ],
    titles=[
        "T1c",
        "T1",
        "Flair",
        "T2"
        ],
    label="Skull stripped",
)

# showcase the defacing result from a more suitable angle
#visualize_defacing(file=t1c_normalized_defaced_output_path)
