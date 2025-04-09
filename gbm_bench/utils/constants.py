from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


# DIRECTORIES
ATLAS_DIR = DATA_DIR / "sri24_atlas"
GROWTH_MODEL_DIR = DATA_DIR / "models"


# SCHEMATA
GROWTH_PRED_SCHEMA = "{subject_id}.nii.gz"
