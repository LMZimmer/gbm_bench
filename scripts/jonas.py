import os   
import argparse
from pathlib import Path
from gbm_bench.utils.constants import RHUH_GBM_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.preprocessing.preprocess import preprocess_exam, process_longitudinal
from gbm_bench.prediction.predict import predict_tumor_growth
from gbm_bench.utils.visualization import plot_model_multislice, plot_recurrence_multislice


if __name__ == "__main__":
    # Example:
    # python scripts/preprocess_rhuh.py -cuda_device 0
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    dcm2niix_location = Path("/home/home/lucas/bin/dcm2niix")  # this binary is executable by all

    # Example input, has to be Path
    test_data_basedir = Path("/home/home/lucas/projects/gbm_bench/test_data/nifti/")
    t1_dir = test_data_basedir / "t1.nii.gz"
    t1c_dir = test_data_basedir / "t1c.nii.gz"
    t2_dir = test_data_basedir / "t2.nii.gz"
    flair_dir = test_data_basedir / "flair.nii.gz"
    tumorseg_dir = test_data_basedir / "tumoresg.nii.gz"
    outdir = Path("./tmp_testdata")

    # Preprocessing (if input is processed niftis, only use tissueseg and manually set folders)
    preprocess_nifti(
            t1_file=Path("test_data/nifti/t1.nii.gz"),
            t1c_file=Path("test_data/nifti/t1c.nii.gz"),
            t2_file=Path("test_data/nifti/t2.nii.gz"),
            flair_file=Path("test_data/nifti/flair.nii.gz"),
            pre_treatment=True,
            outdir=outdir_tmp / "nifti_preop",
            is_coregistered=True,
            tumorseg_file=Path("test_data/nifti/tumorseg.nii.gz"),
            cuda_device=args.cuda_device
            )

    # Inference
    predict_tumor_growth(
            preop_dir=exam["t1"].parent,
            model_id="sbtc",  # spatial brain tumor estimation
            cuda_device=args.cuda_device
            )
