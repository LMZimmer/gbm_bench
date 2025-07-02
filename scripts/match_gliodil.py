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
    segm_paths = sorted(
        p for p in root_path.glob("data_*/*")
        if p.name == "segm.nii.gz" and p.is_file()
    )
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

    # Voxel counts dicom data
    voxel_counts = {}
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

    # Voxel counts ess data
    for ind, e_tseg in enumerate(essential_tumorsegs):
        print(f"Processing {ind}")
        voxel_counts_ess[e_tseg] = voxel_count(e_tseg)

    # Matching
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
            matches[segdir_ess] = candidates[0]
        else:
            no_matches.append(segdir_ess)
            print(f"No candidates found for {segdir_ess}.")

    pprint.pprint(matches, width=80, indent=2)
    print(no_matches)

    # Generate cureated version
    gliodil_new = LongitudinalDataset(dataset_id="GLIODIL", root_dir=gliodil_root)
    gliodil_new.load(GLIODIL_DIR)
    n_keep = 0
    for patient_ind, patient in enumerate(gliodil.patients):
        patient_id = patient["patient_id"]
        for exam in patient["exams"]:
            if exam["timepoint"] != "preop":  # skip postop
                continue
            tumorseg_file = str(exam["tumorseg"])
            if tumorseg_file not in matches.values():
                print(f"No match for {tumorseg_file}, removing patient {patient_id}")
                gliodil_new.remove_patient(patient_id)
            else:
                print(f"Match found, keeping {patient_id}")
                n_keep += 1

    # Save
    outfile = "/home/home/lucas/projects/gbm_bench/gbm_bench/data/datasets/gliodil_subset.json"
    gliodil_new.save(outfile)
    print(n_keep)
