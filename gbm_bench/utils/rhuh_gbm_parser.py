import argparse
import os
import glob
from gbm_bench.preprocessing.preprocess import preprocess_dicom


def rhuh_sort_func(exam_dir):
    date = os.path.basename(exam_dir).replace("-", "")
    date = date[4:8] + date[0:2] + date[2:4]
    return date


def find_modalities(exam_dir):
    sequence_dirs = [d for d in os.listdir(exam_dir) if d != "preprocessing"]
    modalities = {}

    for d in sequence_dirs:
        d_lower = d.lower().replace("-", " ")
        if "flair" in d.lower() and not "t1" in d.lower():
            modalities["flair"] = os.path.join(exam_dir, d)
            continue
        if any([pattern in d_lower for pattern in ["t1 se", "t1 fse", "t1wse", "t1 flair", "92851", "24651", "40131"]]):
            modalities["t1"] = os.path.join(exam_dir, d)
            continue
        if any([pattern in d_lower for pattern in ["dwi", "difusion", "diffusion"]]):
            modalities["diffusion"] = os.path.join(exam_dir, d)
            continue
        if "t2" in d.lower():
            modalities["t2"] = os.path.join(exam_dir, d)
            continue
        if "t1" in d_lower or "3d" in d_lower:
            modalities["t1c"] = os.path.join(exam_dir, d)

    if len(modalities) != 5:
        raise ValueError(f"Only found {len(modalities)} modalities for exam {exam_dir} \n {modalities} \n {os.listdir(exam_dir)}")

    return modalities


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
    # python rhuh_gbm_parser.py -patient_dir /home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM -cuda_device 2
    parser = argparse.ArgumentParser()
    parser.add_argument("-patient_dir", type=str, help="Path to directory containing the patients with exams and DICOM data.")
    parser.add_argument("-cuda_device", type=str, help="GPU id to run on.")
    args = parser.parse_args()
    
    preop_exams = rhuh_parse_exams(args.patient_dir, preop=True)
    print(f"Found {len(preop_exams)} pre-op exams: {preop_exams[0]}...")
    
    #postop_exams = rhuh_parse_exams(args.patient_dir, preop=False)
    #print(f"Found {len(postop_exams)} post-op exams: {postop_exams[0]}...")

    for ind, e in enumerate(preop_exams[2:]):
        print(f"Processing exam {ind}: {e} \n")
        preprocess_dicom(
            t1 = e["t1"],
            t1c = e["t1c"],
            t2 = e["t2"],
            flair = e["flair"],
            cuda_device=args.cuda_device
            )
    
    """
    for e in postop_exams:
        preprocess_dicom(
            t1 = e["t1"],
            t1c = e["t1c"],
            t2 = e["t2"],
            flair = e["flair"],
            cuda_device=args.cuda_device
            )
    """
