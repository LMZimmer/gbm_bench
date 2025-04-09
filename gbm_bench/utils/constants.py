from pathlib import Path
from typing import Any, Union


class PathSchema:
    """
    A simple helper class that wraps a format string to generate Path objects. Accepts both string and Path objects and
    supports the "/" operator to join additional format strings or paths in the same fasion as the Path object.

    Attributes:
        schema (str): The schema string containing placeholders for formatting.
    """
    def __init__(self, schema: Union[str, Path]) -> None:
        if isinstance(schema, Path):
            self.schema = str(schema)
        else:
            self.schema = schema

    def format(self, **kwargs: Any) -> Path:
        """
        Format the stored schema with the given keyword arguments and return a Path object.

        Parameters:
            **kwargs: Keyword arguments used to replace placeholders in the schema string.

        Returns:
            Path: A Path object constructed from the formatted string.
        """
        return Path(self.schema.format(**kwargs))


# DIRECTORIES
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

ATLAS_DIR = DATA_DIR / "sri24_atlas"
GROWTH_MODEL_DIR = DATA_DIR / "models"


# OUTPUT DIRECTORY NAMES
OUTPUT_FOLDER = "preprocessing"
CONVERSION_FOLDER = "nifti_conversion"
SKULL_STRIP_FOLDER = "skull_stripped"
TISSUE_SEGMENTATION_FOLDER = "tissue_segmentation"
TUMOR_SEGMENTATION_FOLDER = "tumor_segmentation"
MODEL_OUTPUT_DIR = "growth_models"


# SCHEMATA
EXAM_BASE_SCHEMA = PathSchema("{exam_dir}") / OUTPUT_FOLDER

BRAIN_MASK_SCHEMA = EXAM_BASE_SCHEMA / SKULL_STRIP_FOLDER / "t1c_bet_mask.nii.gz"
T1_SCHEMA = EXAM_BASE_SCHEMA / SKULL_STRIP_FOLDER / "t1_bet_normalized.nii.gz"
T1C_SCHEMA = EXAM_BASE_SCHEMA / SKULL_STRIP_FOLDER / "t1c_bet_normalized.nii.gz"
T2_SCHEMA = EXAM_BASE_SCHEMA / SKULL_STRIP_FOLDER / "t2_bet_normalized.nii.gz"
FLAIR_SCHEMA = EXAM_BASE_SCHEMA / SKULL_STRIP_FOLDER / "flair_bet_normalized.nii.gz"

TUMORSEG_SCHEMA = EXAM_BASE_SCHEMA / TUMOR_SEGMENTATION_FOLDER / "tumor_seg.nii.gz"
HEALTHY_BRAIN_MASK_SCHEMA = EXAM_BASE_SCHEMA / TUMOR_SEGMENTATION_FOLDER / "healthy_brain_mask.nii.gz"

TISSUE_SEG_SCHEMA = EXAM_BASE_SCHEMA / TISSUE_SEGMENTATION_FOLDER / "tissue_seg.nii.gz"
GM_SCHEMA = EXAM_BASE_SCHEMA / TISSUE_SEGMENTATION_FOLDER / "gm.nii.gz"
WM_SCHEMA = EXAM_BASE_SCHEMA / TISSUE_SEGMENTATION_FOLDER / "wm.nii.gz"
CSF_SCHEMA = EXAM_BASE_SCHEMA / TISSUE_SEGMENTATION_FOLDER / "csf.nii.gz"
GM_PBMAP_SCHEMA = EXAM_BASE_SCHEMA / TISSUE_SEGMENTATION_FOLDER / "gm_pbmap.nii.gz"
WM_PBMAP_SCHEMA = EXAM_BASE_SCHEMA / TISSUE_SEGMENTATION_FOLDER / "wm_pbmap.nii.gz"
CSF_PBMAP_SCHEMA = EXAM_BASE_SCHEMA / TISSUE_SEGMENTATION_FOLDER / "csf_pbmap.nii.gz"

GROWTH_PRED_SCHEMA = "{subject_id}.nii.gz"
