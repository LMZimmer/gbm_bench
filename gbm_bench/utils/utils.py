import os
import ants
import datetime
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PyPDF2 import PdfMerger
from typing import List, Tuple, Optional
from scipy.ndimage import center_of_mass


def compute_center_of_mass(seg_data: np.ndarray, mri_data: np.ndarray, classes: List[int] = [1, 2, 3],) -> Tuple[int, int, int]:

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


def load_and_resample_mri_data(filepath: str, resample_params: Tuple[int, int, int], interp_type: Optional[int] = 0,) -> np.ndarray:
    
    img = ants.image_read(filepath)
    img = ants.resample_image(
            image=img,
            resample_params=resample_params,
            use_voxels=True,
            interp_type=interp_type
            )
    return img.to_nibabel().get_fdata()


def merge_pdfs(pdf_list: List[str], output_pdf: str) -> None:
    pdf_merger = PdfMerger()

    for pdf in pdf_list:
        pdf_merger.append(pdf)

    pdf_merger.write(output_pdf)
    pdf_merger.close()
    print(f"Combined PDF saved as {output_pdf}")


def timed_print(print_message: str) -> None:
    time = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[INFO | {time}]: ", print_message)
