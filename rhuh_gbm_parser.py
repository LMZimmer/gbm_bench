import argparse
import os
import glob
from preprocess import preprocess_dicom


# disregarding capitalization
# T1  sequences: T1+SE, T1+FSE              -> T1 spin echo
# T1c sequences: T1+3d, sag+SPGR+3d, T1-FFE-PROSET-GD SENSE         -> 3d + some contrast agent
# T2  sequences: T2+FRFSE, T2+TSE, T2+FSE   -> T2 spin echo
# FLAIR sequenc: FLAIR
# DIFF  seqienc: DIFUSION, DWI, DWI

# weird ass cases
# t1: 6.000000-Obl T1 3D FSPGR IR-92851
# t1c: 9.000000-3D Ax BRAVO C-92734

#ValueError: Only found 3 modalities for exam /home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0034/07-20-2014-NA-RM DE CRANEO SINCON CONTRASTE-27430 
# {'t1c': '13.000000-T1 3D VIBE AXIAL  avant gado -24651', 'flair': '11.000000-T2 FLAIR 3D SAG 1.25 FS-39663', 'diffusion': '5.000000-DIFFUSION EPI 3bTRACEWDFC-52607'} 
# ['14.000000-T1 3D VIBE AXIAL FS  GADO-14660', '11.000000-T2 FLAIR 3D SAG 1.25 FS-39663', '5.000000-DIFFUSION EPI 3bTRACEWDFC-52607', '13.000000-T1 3D VIBE AXIAL  avant gado -24651', '12.000000-T2 EG TRA 256 NEW-55362']


def rhuh_sort_func(exam_dir):
    date = os.path.basename(exam_dir).replace("-","")
    date = date[4:8] + date[0:2] + date[2:4]
    return date


def find_modalities(exam_dir):
    sequence_dirs = os.listdir(exam_dir)
    modalities = {}

    for d in sequence_dirs:
        d_lower = d.lower().replace("-", " ")
        if "t1 se" in d_lower or "t1 fse" in d_lower or "92851" in d_lower:
            modalities["t1"] = d
            continue
        if "flair" in d.lower():
            modalities["flair"] = d
            continue
        if "t2" in d.lower() and "se" in d.lower():
            modalities["t2"] = d
            continue
        if "difusion" in d.lower() or "diffusion" in d.lower() or "dwi" in d.lower():
            modalities["diffusion"] = d
            continue
        if "t1" in d_lower or "3d" in d_lower:
            modalities["t1c"] = d

    print(modalities)
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
    # python rhuh_gbm_parser.py -patient_dir /home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM -gpu_device 2
    parser = argparse.ArgumentParser()
    parser.add_argument("-patient_dir", type=str, help="Path to directory containing the patients with exams and DICOM data.")
    parser.add_argument("-gpu_device", type=str, help="GPU id to run on.")
    args = parser.parse_args()
    
    preop_exams = rhuh_parse_exams(args.patient_dir, preop=True)

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
