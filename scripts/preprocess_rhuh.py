import os   
import argparse
from pathlib import Path
from gbm_bench.utils.parsing import RHUHParser
from gbm_bench.preprocessing.preprocess import preprocess_dicom, process_longitudinal

if __name__ == "__main__":
    # Example:
    # python scripts/preprocess_rhuh.py -cuda_device 4
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="4", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Define directories
    rhuh_root = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
    dcm2niix_location = "/home/home/lucas/bin/dcm2niix"

    # Collect patient, exam, image paths
    rhuh_parser = RHUHParser(root_dir=rhuh_root)
    rhuh_parser.parse()
    patients = rhuh_parser.get_patients()

    # Process individual exams
    for patient_ind, patient in enumerate(patients):
        print(f"Processing {patient_ind}/{len(patients)}...")
        
        for exam_ind, sequences in enumerate(patient["sequences"]):
            print(f"Exam {exam_ind}...")

            # Skip postop, only process preop and follow up
            if exam_ind==1:
                continue

            # Preop is 0, follow up is 2
            is_preop = True if exam_ind==0 else False

            exam_dir = Path(os.path.dirname(sequences["t1"]))

            preprocess_dicom(
                    t1=Path(sequences["t1"]),
                    t1c=Path(sequences["t1c"]),
                    t2=Path(sequences["t2"]),
                    flair=Path(sequences["flair"]),
                    outdir=exam_dir,
                    dcm2niix_location=Path(dcm2niix_location),
                    pre_treatment=is_preop,
                    cuda_device=args.cuda_device,
                    perform_nifti_conversion=True,
                    perform_skullstripping=True,
                    perform_tumorseg=True,
                    perform_tissueseg=is_preop
                    )

    # Longitudinal registration (preop exam and exam 2)
    for patient_ind, patient in enumerate(patients):
        print(f"Performing longitudinal registration {patient_ind}/{len(patients)}: {patient['exams'][2]}")

        process_longitudinal(
                preop_exam=Path(patient["exams"][0]),
                postop_exam=Path(patient["exams"][2]),
                outdir=Path(patient["exams"][0])
                )
