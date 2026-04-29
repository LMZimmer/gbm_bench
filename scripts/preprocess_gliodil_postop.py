import os   
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
from gbm_bench.utils.constants import GLIODIL_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.preprocessing.preprocess import preprocess_nifti, process_longitudinal


def convert_tumorseg_labels(inpath, outpath):
    img = nib.load(str(inpath))
    data = np.rint(img.get_fdata()).astype(np.int32)

    data[data == 4] = 3

    new_img = nib.Nifti1Image(data, affine=img.affine, header=img.header)
    nib.save(new_img, str(outpath))


if __name__ == "__main__":
    # Example:
    # python scripts/preprocess_gliodil_postop.py -cuda_device 1
    # nohup python -u scripts/preprocess_gliodil_postop.py -cuda_device 1 > tmp_gliodil_preproc_postop.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Read dataset
    gliodil_root = "/mnt/Drive2/lucas/datasets/GLIODIL"
    gliodil = LongitudinalDataset(dataset_id="GLIODIL", root_dir=gliodil_root)
    gliodil.load(GLIODIL_DIR)

    """
    # Individual exams
    for patient_ind, patient in enumerate(gliodil.patients):
        print(f"Processing {patient_ind}/{len(gliodil.patients)}...")

        patient_id = patient["patient_id"]

        for exam in patient["exams"]:
            if exam["timepoint"] != "postop":  # only postop
                continue

            is_preop = (exam["timepoint"] == "preop")
            print(f"{exam['t1c']}")

            patient_dir = exam["t1c"].parent
            outdir = patient_dir / f"{'preop' if is_preop else 'postop'}"
            outdir.mkdir(exist_ok=True)

            try:
                preprocess_nifti(
                        t1_file=exam["t1"],
                        t1c_file=exam["t1c"],
                        t2_file=exam["t2"],
                        flair_file=exam["flair"],
                        pre_treatment=is_preop,
                        outdir=outdir,
                        is_skull_stripped=True,
                        is_coregistered=True,
                        cuda_device=args.cuda_device
                        )
            except Exception as e:
                print(f"Exception {e} for patient {patient_id}")
    """

    # Longitudinal registration
    for patient_ind, patient in enumerate(gliodil.patients):
        print(f"Performing longitudinal registration {patient_ind}/{len(gliodil.patients)}.")

        patient_id = patient["patient_id"]
        preop_exam = gliodil.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]  # Find first preop exam
        preop_exam_dir = preop_exam["t1c"].parent / "preop"

        # Loop through followup exams
        followup_exams = gliodil.get_patient_exams(patient_id=patient_id, timepoint="postop") # changed to postop
        
        for followup_exam in followup_exams:
            followup_exam_dir = followup_exam["t1c"].parent / "postop"
            
            try:
                process_longitudinal(
                        preop_exam_dir=preop_exam_dir,
                        followup_exam_dir=followup_exam_dir,
                        outdir=followup_exam_dir,
                        is_coregistered=False
                        )
            except Exception as e:
                print(f"Exception {e} for patient {patient_id}")
    
    print(f"Finished processing.")
