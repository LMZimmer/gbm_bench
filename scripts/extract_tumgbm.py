import os
import json
import shutil
import argparse
import numpy as np
import nibabel as nib
from scipy import stats
from pathlib import Path
from gbm_bench.utils.utils import merge_pdfs, load_mri_data
from gbm_bench.utils.constants import *
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.utils.visualization import plot_tumor_sizes, plot_performances, plot_com_distances


if __name__ == "__main__":
    # Example:
    # python scripts/extract_tumgbm.py -outdir /mnt/Drive2/lucas/tum_gbm
    parser = argparse.ArgumentParser()
    parser.add_argument("-outdir", type=str, help="Output directory.")
    args = parser.parse_args()

    DATASET_IDS = ["GLIODIL"]  # ["TCGA-LGG"]
    DATASET_DIRS = [GLIODIL_DIR]  # [TCGA_LGG_DIR]
    ROOT_DIRS = [
        "/mnt/Drive2/lucas/datasets/GLIODIL",
        # "/mnt/Drive2/lucas/datasets/TCGA-LGG"
    ]

    for d_id, d_dir, d_rootdir in zip(DATASET_IDS, DATASET_DIRS, ROOT_DIRS):
        print(f"Evaluating {d_id}...")

        dataset = LongitudinalDataset(dataset_id=d_id, root_dir=d_rootdir)
        dataset.load(d_dir)

        for patient in dataset.patients:
            patient_id = patient["patient_id"]

            preops = dataset.get_patient_exams(patient_id=patient_id, timepoint="preop")
            followups = dataset.get_patient_exams(patient_id=patient_id, timepoint="followup")
            postops = dataset.get_patient_exams(patient_id=patient_id, timepoint="postop")

            if not preops:
                continue  # require at least preop

            preop_exam = preops[0]
            followup_exam = followups[0] if followups else None
            postop_exam = postops[0] if postops else None

            if "UPENN" in d_id:
                preop_exam_dir = preop_exam["t1"].parent
                followup_exam_dir = followup_exam["t1"].parent if followup_exam else None
                postop_exam_dir = postop_exam["t1"].parent if postop_exam else None

            elif "GLIODIL" in d_id:
                no_t1c = [
                    "data_001", "data_013", "data_020", "data_030", "data_034",
                    "data_991", "data_992", "data_994", "data_995", "data_998",
                ]
                if patient_id in no_t1c:
                    continue

                preop_exam_dir = preop_exam["t1c"].parent / "preop"
                followup_exam_dir = (
                    followup_exam["t1c"].parent / "followup" if followup_exam else None
                )
                postop_exam_dir = (
                    postop_exam["t1c"].parent / "postop" if postop_exam else None
                )

            else:
                preop_exam_dir = preop_exam["t1c"].parent
                followup_exam_dir = (
                    followup_exam["t1c"].parent if followup_exam else None
                )
                postop_exam_dir = (
                    postop_exam["t1c"].parent if postop_exam else None
                )

            try:
                # Build timepoint → directory mapping (only existing ones)
                exams = {"preop": preop_exam_dir}
                if followup_exam_dir is not None:
                    exams["followup"] = followup_exam_dir
                if postop_exam_dir is not None:
                    exams["postop"] = postop_exam_dir

                patient_outdir = Path(args.outdir) / patient_id
                patient_outdir.mkdir(parents=True, exist_ok=True)

                for tp, base_dir in exams.items():
                    tp_outdir = patient_outdir / tp
                    tp_outdir.mkdir(parents=True, exist_ok=True)

                    for modality in ["t1c", "t1", "t2", "flair"]:
                        cf = MODALITY_STRIPPED_SCHEMA.format(
                            base_dir=base_dir, modality=modality
                        )
                        if cf.is_file():
                            shutil.copy(str(cf), str(tp_outdir / cf.name))

            except Exception as e:
                shutil.rmtree(patient_outdir, ignore_errors=True)
                print(f"Exception for {patient_id}: {e}")
