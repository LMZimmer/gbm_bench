import os   
import argparse
import numpy as np
import nibabel as nib
import pprint
from pathlib import Path
from gbm_bench.utils.constants import GLIODIL_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.preprocessing.preprocess import preprocess_nifti, process_longitudinal


def collect_segm_maps(root_dir):
    root_path = Path(root_dir).expanduser().resolve()

    # Pattern: one level down, file must be called exactly “segm.nii.gz”.
    segm_paths = sorted(
        p for p in root_path.glob("data_*/*")
        if p.name == "segm.nii.gz" and p.is_file()
    )

    # Convert to strings so the result is JSON-serialisable / CLI-friendly.
    return [str(p) for p in segm_paths]


def voxel_count(seg):
    data = nib.load(str(seg)).get_fdata()
    mask = (data != 0)
    return int(np.sum(mask))


if __name__ == "__main__":
    # Example:
    # python scripts/match_gliodil.py

    # Read dataset
    gliodil_root = "/mnt/Drive2/lucas/datasets/GLIODIL"
    gliodil = LongitudinalDataset(dataset_id="GLIODIL", root_dir=gliodil_root)
    gliodil.load(GLIODIL_DIR)

    voxel_counts = {}
    # Individual exams
    for patient_ind, patient in enumerate(gliodil.patients):
        print(f"Processing {patient_ind}/{len(gliodil.patients)}...")

        for exam in patient["exams"]:
            if exam["timepoint"] != "preop":  # skip postop
                continue

            tumorseg_file = str(exam["tumorseg"])
            voxel_counts[tumorseg_file] = voxel_count(tumorseg_file)

    essential_root_dir = "/mnt/Drive2/lucas/datasets/data_GliODIL_essential"
    essential_tumorsegs = collect_segm_maps(essential_root_dir)
    voxel_counts_ess = {}

    for ind, e_tseg in enumerate(essential_tumorsegs):
        print(f"Processing {ind}")
        voxel_counts_ess[e_tseg] = voxel_count(e_tseg)

    matches = {}
    no_matches = []
    for segdir_ess, vcount_ess in voxel_counts_ess.items():
        candidates = []
        for segdir, vcount in voxel_counts.items():
            if vcount == vcount_ess:
                print(f"Same voxel count {vcount_ess} / {vcount} for {segdir_ess} / {segdir}.")
                candidates.append(segdir)
        if len(candidates)>1:
            print(f"More than one match for {segdir_ess}.")
        elif len(candidates)==1:
            matches[segdir_ess] = segdir
        else:
            no_matches.append(segdir_ess)
            print(f"No candidates found for {segdir_ess}.")

    pprint.pprint(matches, width=80, indent=2)
    print(no_matches)
