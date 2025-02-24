import os   
import argparse
from gbm_bench.utils.parsing import RHUHParser
from gbm_bench.preprocessing.preprocess import preprocess_dicom, process_longitudinal

if __name__ == "__main__":
    # Example:
    # python scripts/preprocess_rhuh.py -cuda_device 1
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="1", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    rhuh_root = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
    dcm2niix_location = "/home/home/lucas/bin/dcm2niix"

    rhuh_parser = RHUHParser(root_dir=rhuh_root)
    rhuh_parser.parse()
    patients = rhuh_parser.get_patients()


    # Process exams
    for patient_ind, patient in enumerate(patients):
        print(f"Processing {patient_ind}/{len(patients)}...")
        
        for exam_ind, sequences in enumerate(patient["sequences"][1:]): #postop only
            print(f"Exam {exam_ind}...")

            # Exams are sorted, first one is pre-op for RHUH
            is_preop = True if exam_ind==0 else False

            if os.path.exists(os.path.join(os.path.dirname(sequences["t1"]), "preprocessing")):
                continue

            preprocess_dicom(
                    t1=sequences["t1"],
                    t1c=sequences["t1c"],
                    t2=sequences["t2"],
                    flair=sequences["flair"],
                    dcm2niix_location=dcm2niix_location,
                    pre_treatment=is_preop,
                    cuda_device=args.cuda_device,
                    perform_nifti_conversion=True,
                    perform_skullstripping=True,
                    perform_tumorseg=True,
                    perform_tissueseg=is_preop
                    )

    # Longitudinal registration (preop exam and exam 2)
    """
    for patient_ind, patient in enumerate(patients):
        print(f"Performing longitudinal registration {patient_ind}/{len(patients)}...")
        
        process_longitudinal(
                preop_exam=patient["exams"][0],
                postop_exam=patient["exams"][2]
                )
    """
