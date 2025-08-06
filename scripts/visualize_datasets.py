import os
import shutil
import argparse
from pathlib import Path
from gbm_bench.utils.utils import merge_pdfs
from gbm_bench.utils.constants import *
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.utils.visualization import plot_tumor_sizes, plot_performances, plot_com_distances, plot_diff_vs_distance, plot_missed


if __name__ == "__main__":
    # Example:
    # python scripts/visualize_datasets.py

    DATASET_IDS = [
            "RHUH",
            #"UPENN",
            "LUMIERE",
            "GLIODIL",
            #"IVYGAP",
            #"CPTAC",
            #"TCGA-GBM"
            ]#, "TCGA-LGG"]
    DATASET_DIRS = [
            RHUH_GBM_DIR,
            #UPENN_GBM_DIR,
            LUMIERE_DIR,
            GLIODIL_DIR,
            #IVYGAP_DIR,
            #CPTAC_DIR,
            #TCGA_GBM_DIR
            ]#, TCGA_LGG_DIR]
    ROOT_DIRS = [
            "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM",
            #"/home/home/lucas/data/UPENN-GBM/UPENN-GBM",
            "/mnt/Drive2/lucas/datasets/LUMIERE/Imaging",
            "/mnt/Drive2/lucas/datasets/GLIODIL",
            #"/mnt/Drive2/lucas/datasets/IVYGAP",
            #"/mnt/Drive2/lucas/datasets/CPTAC-GBM",
            #"/mnt/Drive2/lucas/datasets/TCGA-GBM",
            #"/mnt/Drive2/lucas/datasets/TCGA-LGG"
            ]

    model_id = "gliodil"
    outfile_tsize = "/home/home/lucas/projects/gbm_bench/tmp/tumor_sizes.pdf"
    outfile_rsize = "/home/home/lucas/projects/gbm_bench/tmp/recurrence_sizes.pdf"
    outfile_perf = f"/home/home/lucas/projects/gbm_bench/tmp/performances_{model_id}.pdf"
    outfile_dist = "/home/home/lucas/projects/gbm_bench/tmp/com_distances.pdf"
    outfile_diff_dist = "/home/home/lucas/projects/gbm_bench/tmp/distance_vs_difference.pdf"
    outfile_missed_volume = f"/home/home/lucas/projects/gbm_bench/tmp/missed_volume_{model_id}.pdf"

    """
    plot_tumor_sizes(
            dataset_ids=DATASET_IDS,
            dataset_dirs=DATASET_DIRS,
            dataset_rootdirs=ROOT_DIRS,
            outfile=outfile_tsize
            )

    plot_tumor_sizes(
            dataset_ids=DATASET_IDS,
            dataset_dirs=DATASET_DIRS,
            dataset_rootdirs=ROOT_DIRS,
            outfile=outfile_rsize,
            recurrence=True
            )
    
    """
    plot_com_distances(
            dataset_ids=DATASET_IDS,
            dataset_dirs=DATASET_DIRS,
            dataset_rootdirs=ROOT_DIRS,
            outfile=outfile_dist
            )

    """
    plot_performances(
            dataset_ids=DATASET_IDS,
            dataset_dirs=DATASET_DIRS,
            dataset_rootdirs=ROOT_DIRS,
            model_id=model_id,
            outfile=outfile_perf
            )

    plot_diff_vs_distance(
            dataset_ids=DATASET_IDS,
            dataset_dirs=DATASET_DIRS,
            dataset_rootdirs=ROOT_DIRS,
            model_id=model_id,
            outfile=outfile_diff_dist
            )

    plot_missed(
            dataset_ids=DATASET_IDS,
            dataset_dirs=DATASET_DIRS,
            dataset_rootdirs=ROOT_DIRS,
            model_id=model_id,
            outfile=outfile_missed_volume
            )
    """
