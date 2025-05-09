import os   
import argparse
from pathlib import Path
from gbm_bench.utils.constants import GLIODIL_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.preprocessing.preprocess import preprocess_nifti, process_longitudinal


if __name__ == "__main__":
    # Example:
    # python scripts/preprocess_gliodil1.py -cuda_device 2
    # nohup python -u scripts/preprocess_gliodil1.py -cuda_device 2 > tmp_gliodil_preproc1.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Read dataset
    gliodil_root = "/mnt/Drive2/lucas/datasets/GLIODIL"
    gliodil = LongitudinalDataset(dataset_id="GLIODIL", root_dir=gliodil_root)
    gliodil.load(GLIODIL_DIR)

    # Individual exams
    for patient_ind, patient in enumerate(gliodil.patients[80:]):  #started 80
        print(f"Processing {patient_ind}/{len(gliodil.patients)}...")

        for exam in patient["exams"]:
            if exam["timepoint"] != "preop":  # skip postop and followup, theyre already registered
                continue

            is_preop = (exam["timepoint"] == "preop")
            print(f"{exam['t1']}")

            preprocess_nifti(
                    t1_file=exam["t1"],
                    t1c_file=exam["t1c"],
                    t2_file=exam["t2"],
                    flair_file=exam["flair"],
                    pre_treatment=is_preop,
                    outdir=exam["t1"].parent,
                    is_skull_stripped=True,
                    is_coregistered=True,
                    cuda_device=args.cuda_device
                    )

    # Longitudinal registration
    """
    for patient_ind, patient in enumerate(gliodil.patients[60:]):
        print(f"Performing longitudinal registration {patient_ind}/{len(gliodil.patients)}.")

        patient_id = patient["patient_id"]
        preop_exam = gliodil.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]  # Find first preop exam
        preop_exam_dir = preop_exam["t1"].parent

        # Loop through followup exams
        followup_exams = gliodil.get_patient_exams(patient_id=patient_id, timepoint="followup")
        
        for followup_exam in followup_exams:
            followup_exam_dir = followup_exam["t1"].parent
            
            process_longitudinal(
                    preop_exam_dir=preop_exam_dir,
                    followup_exam_dir=followup_exam_dir,
                    outdir=followup_exam_dir
                    )
    
    print(f"Finished processing.")
    """
