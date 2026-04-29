import os
import argparse
from pathlib import Path
from gbm_bench.utils.constants import GLIODIL_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.prediction.predict import predict_tumor_growth


if __name__ == "__main__":
    # Example:
    # python scripts/predict_gliodil.py -cuda_device 0
    # nohup python -u scripts/predict_gliodil.py -cuda_device 3 > tmp_gliodil_pred.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    # Read dataset
    gliodil_root = "/mnt/Drive2/lucas/datasets/GLIODIL"
    gliodil = LongitudinalDataset(dataset_id="GLIODIL", root_dir=gliodil_root)
    gliodil.load(GLIODIL_DIR)

    remaining = ["respond_tum_130"]

    # Predict on preop exams
    for patient_ind, patient in enumerate(gliodil.patients):
        patient_id = patient["patient_id"]
        print(f"Predicting {patient_ind}/{len(gliodil.patients)}...")

        if patient_id not in remaining:
            continue

        for exam in patient["exams"]:
            if exam["timepoint"] != "preop":
                continue

            print(exam["t1c"].parent)
            preop_dir = exam["t1c"].parent / "preop"

            predict_tumor_growth(
                    preop_dir=preop_dir,
                    model_id="sbtc", # lmi, sbtc, gliodil
                    cuda_device=args.cuda_device
                    )
    print("Done.")
