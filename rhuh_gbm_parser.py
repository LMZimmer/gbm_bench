import argparse
import os
import glob
from preprocess import preprocess_dicom


SEQUENCE_PATTERNS = {
        "t1" : ["*T1 SE*", "*T1-SE*", "*T1 FSE*", "*T1-FSE*"],
        "t1c" : ["*T1 3D*", "*T1-3D*", "*T1 3d*", "*T1 FFE*", "*T1-FFE*"],
        "t2" : ["*T2 FRFSE*", "*T2-FRFSE*", "*T2 TSE*", "*T2-TSE*", "*T2 FSE*", "*T2-FSE*", "*FRFSE T2*"],
        "flair" : ["*FLAIR*"],
        "diffusion": ["*DIFUSION*", "*DWI*", "*Difusion*"]
        }


def rhuh_sort_func(exam_dir):
    date = os.path.basename(exam_dir).replace("-","")
    date = date[4:8] + date[0:2] + date[2:4]
    return date


def find_modalities(exam_dir):
    sequences = {}
    for seq, pattern in SEQUENCE_PATTERNS.items():
        matches = []
        for p in pattern:
            match = glob.glob(os.path.join(exam_dir, p))
            matches = matches + match
        print(exam_dir, pattern, matches)
        if len(matches)>1:
            raise ValueError(f"Found more than one sequence in {exam_dir} matching {pattern}.")
        if len(matches)==0:
            raise ValueError(f"Found no sequence in {exam_dir} matching {pattern}.")
    return matches[0]


def rhuh_get_modality(sequence_dir):
    sequences = [f.path for f in os.scandir(sequence_dir) if f.is_dir()]


def rhuh_parse_exams(patient_dir, preop):
    # Parse patients
    patients = glob.glob(os.path.join(patient_dir, "RHUH-*"))
    # Parse exams, sort by date, return first for preop and later exams for postop
    exams = []
    for p in patients:
        patient_exams = glob.glob(os.path.join(p, "*-NA-*"))
        patient_exams.sort(key=rhuh_sort_func)
        if preop:
            exams.append(find_modalities(patient_exams[0]))
        else:
            exams = exams + [find_modalities(p) for p in patient_exams[1:]]
    return exams


if __name__=="__main__":
    # Example
    # python rhuh_gbm_parser.py -patient_dir /home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM -gpu_device 2
    parser = argparse.ArgumentParser()
    parser.add_argument("-patient_dir", type=str, help="Path to directory containing the patients with exams and DICOM data.")
    parser.add_argument("-gpu_device", type=str, help="GPU id to run on.")
    args = parser.parse_args()
    
    preop_exams = rhuh_parse_exams(args.patient_dir, preop=True)
    print(preop_exams)

    """
    for e in preop_exams:
        preprocess_dicom(
            t1 = os.path.join(e, ""),
            t1c = os.path.join(e, ""),
            t2 = os.path.join(e, ""),
            flair = os.path.join(e, ""),
            gpu_device=args.gpu_device
            )
    """

