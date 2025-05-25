import os   
import argparse
import nibabel as nib
from pathlib import Path
from gbm_bench.utils.constants import GLIODIL_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.preprocessing.preprocess import preprocess_nifti, process_longitudinal
from gbm_bench.prediction.predict import predict_tumor_growth
from gbm_bench.evaluation.evaluate import evaluate_tumor_model
from gbm_bench.utils.constants import PREDICTION_OUTPUT_SCHEMA


def convert_tumorseg_labels(inpath, outpath):
    img = nib.load(str(inpath))
    data = img.get_fdata()

    data[data == 4] = 3

    new_img = nib.Nifti1Image(data, affine=img.affine, header=img.header)
    nib.save(new_img, str(outpath))


if __name__ == "__main__":
    # Example:
    # python scripts/single_nifti.py -cuda_device 1
    # nohup python -u scripts/single_nifti.py -cuda_device 1 > tmp_single_nifti.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Read dataset
    {
      "patient_id": "tgm001",
      "patient_dir": "/Users/cherubim/Desktop/GLIODIL/tgm001",
      "exams": [
        {
          "flair": "/Users/cherubim/Desktop/GLIODIL/tgm001/preop/sub-tgm001_ses-preop_space-sri_flair.nii.gz",
          "t1": "/Users/cherubim/Desktop/GLIODIL/tgm001/preop/sub-tgm001_ses-preop_space-sri_t1.nii.gz",
          "t1c": "/Users/cherubim/Desktop/GLIODIL/tgm001/preop/sub-tgm001_ses-preop_space-sri_t1c.nii.gz",
          "t2": "",
          "tumorseg": "/Users/cherubim/Desktop/GLIODIL/tgm001/preop/sub-tgm001_ses-preop_space-sri_seg.nii.gz",
          "timepoint": "preop"
        },
        {
          "t1c": "/Users/cherubim/Desktop/GLIODIL/tgm001/preop/sub-tgm001_ses-preop_space-sri_t1c-rec.nii.gz",
          "tumorseg": "/Users/cherubim/Desktop/GLIODIL/tgm001/preop/sub-tgm001_ses-preop_space-sri_seg-rec.nii.gz",
          "timepoint": "followup"
        }

    patient_id = "tgm001"
    algo_id ="sbtc"

    t1_file = Path("/Users/cherubim/Desktop/GLIODIL/tgm001/preop/sub-tgm001_ses-preop_space-sri_t1.nii.gz")
    t1c_file = Path("/Users/cherubim/Desktop/GLIODIL/tgm001/preop/sub-tgm001_ses-preop_space-sri_t1c.nii.gz")
    t2_file = Path("")
    flair_file = Path("/Users/cherubim/Desktop/GLIODIL/tgm001/preop/sub-tgm001_ses-preop_space-sri_flair.nii.gz")
    tumorseg_file = Path("/Users/cherubim/Desktop/GLIODIL/tgm001/preop/sub-tgm001_ses-preop_space-sri_seg.nii.gz")
    t1c_followup_file = Path("/Users/cherubim/Desktop/GLIODIL/tgm001/preop/sub-tgm001_ses-preop_space-sri_t1c-rec.nii.gz")
    recurrenceseg_file = Path("/Users/cherubim/Desktop/GLIODIL/tgm001/preop/sub-tgm001_ses-preop_space-sri_seg-rec.nii.gz")
    exam_dir_preop = Path("/Users/cherubim/Desktop/GLIODIL/tgm001/preop/preop")
    exam_dir_followup = Path("/Users/cherubim/Desktop/GLIODIL/tgm001/preop/followup")
    
    # Convert tumor segmentations
    converted_tumorseg_file_preop = exam_dir_preop / "tumorseg_123.nii.gz"
    converted_tumorseg_file_followup = exam_dir_followup / "tumorseg_123.nii.gz"
    convert_tumorseg_labels(tumorseg_file, converted_tumorseg_file_preop)
    convert_tumorseg_labels(recurrenceseg_file, converted_tumorseg_file_followup)

    # Preprocessing
    preprocess_nifti(
        t1_file=t1_file,
        t1c_file=t1c_file,
        t2_file=t2_file,
        flair_file=flair_file,
        tumorseg_file=converted_tumorseg_file_preop,
        pre_treatment=True,
        outdir=exam_dir_preop,
        is_skull_stripped=True,
        is_coregistered=True,
        cuda_device=args.cuda_device
        )

    preprocess_nifti(
        t1_file=Path(""),
        t1c_file=t1c_followup_file,
        t2_file=Path(""),
        flair_file=Path(""),
        tumorseg_file=converted_tumorseg_file_followup,
        pre_treatment=False,
        outdir=exam_dir_followup,
        is_skull_stripped=True,
        is_coregistered=True,
        cuda_device=args.cuda_device
        )

    # Longitudinal
    process_longitudinal(
            preop_exam_dir=exam_dir_preop,
            followup_exam_dir=exam_dir_followup,
            outdir=t1c_followup_file
            )

    # Predict
    predict_tumor_growth(
            preop_dir=exam_dir_preop,
            model_id=algo_id,
            cuda_device=args.cuda_device
            )

    # Evaluate
    prediction_dir = PREDICTION_OUTPUT_SCHEMA.format(base_dir=exam_dir_preop, algo_id=algo_id)
    results = evaluate_tumor_model(
            preop_dir=exam_dir_preop,
            followup_dir=exam_dir_followup,
            pred_file=prediction_dir,
            model_id=algo_id
            )
