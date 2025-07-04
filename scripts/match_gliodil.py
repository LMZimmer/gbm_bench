import os   
import argparse
import numpy as np
import nibabel as nib
import pprint
from pathlib import Path
from gbm_bench.utils.constants import GLIODIL_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.preprocessing.preprocess import preprocess_nifti, process_longitudinal


if __name__ == "__main__":
    # Example:
    # python scripts/match_gliodil.py

    # Read dataset
    gliodil_root = "/mnt/Drive2/lucas/datasets/GLIODIL"
    gliodil = LongitudinalDataset(dataset_id="GLIODIL", root_dir=gliodil_root)
    gliodil.load(GLIODIL_DIR)

    rootdir_ess = Path("/mnt/Drive2/lucas/datasets/data_GliODIL_essential")
    ids_ess = []
    for folder in sorted(rootdir_ess.glob("data_*")):
        case_id = str(folder).split("_")[-1]
        if folder.is_dir():
            if case_id.startswith("9"):
                continue
            if case_id.startswith("0"):
                continue
            if case_id.startswith("4"):
                ids_ess.append("respond_tum_0" + case_id[1:])
            if case_id.startswith("5"):
                ids_ess.append("respond_tum_1" + case_id[1:])
            if case_id.startswith("7"):
                ids_ess.append("tgm0" + case_id[1:])
    print(f"Found the {len(ids_ess)} ids in essential data, {len(np.unique(ids_ess))} uniques: {ids_ess}")

    # Generate cureated version
    gliodil_new = LongitudinalDataset(dataset_id="GLIODIL", root_dir=gliodil_root)
    gliodil_new.load(GLIODIL_DIR)
    kept = []
    removed = []
    for patient_ind, patient in enumerate(gliodil.patients):
        patient_id = patient["patient_id"]
        if patient_id not in ids_ess:
            gliodil_new.remove_patient(patient_id)
            removed.append(patient_id)
        else:
            kept.append(patient_id)
            ids_ess.remove(patient_id)

    # Save
    outfile = "/home/home/lucas/projects/gbm_bench/gbm_bench/data/datasets/gliodil_subset.json"
    gliodil_new.save(outfile)
    ids_ess.sort()
    print(len(removed))
    print(len(kept))
    print(removed)
    print(kept)
    print(ids_ess)
