import os
import argparse
from gbm_bench.utils.parsing import RHUHParser
from gbm_bench.preprocessing.preprocess import preprocess_dicom


if __name__ == "__main__":
    # Example:
    # python scripts/preprocess_rhuh.py -cuda_device 1
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="1", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    rhuh_root = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
    dcm2niix_location = "/home/home/lucas/bin/dcm2niix"
    atlas_t1_dir = "/home/home/lucas/bin/miniconda3/envs/brainles/lib/python3.10/site-packages/brainles_preprocessing/registration/atlas/t1_skullstripped_brats_space.nii"
    atlas_tissues_dir = "/home/home/lucas/data/ATLAS/SRI-24/tissues.nii"

    rhuh_parser = RHUHParser(root_dir=rhuh_root)
    rhuh_parser.parse()
    patients = rhuh_parser.get_patients()

    for patient_ind, patient in enumerate(patients):
        print(f"Processing {patient_ind}/{len(patients)}...")
        
        for exam_ind, sequences in enumerate(patient.sequences):
            print(f"Exam {exam_ind}...")
            
            # Exams are sorted, first one is pre-op for RHUH
            pre_treatement = True if exam_ind==0 else False
            preprocess_dicom(
                    t1=sequences.t1,
                    t1c=sequences.t1c,
                    t2=sequences.t2,
                    flair=sequences.flair
                    dcm2niix_location=dcm2niix_location,
                    atlas_t1_dir=atlas_t1_dir,
                    atlas_tissues_dir=atlas_tissues_dir,
                    pre_treatment=pre_treatement,
                    cuda_device=args.cuda_device
                    )
