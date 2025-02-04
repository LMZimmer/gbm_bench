import argparse
import os
import glob
from preprocess import preprocess_dicom

#ValueError: Only found 3 modalities for exam /home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0034/07-20-2014-NA-RM DE CRANEO SINCON CONTRASTE-27430 
# {'t1c': '13.000000-T1 3D VIBE AXIAL  avant gado -24651', 'flair': '11.000000-T2 FLAIR 3D SAG 1.25 FS-39663', 'diffusion': '5.000000-DIFFUSION EPI 3bTRACEWDFC-52607'} 
# ['14.000000-T1 3D VIBE AXIAL FS  GADO-14660', '11.000000-T2 FLAIR 3D SAG 1.25 FS-39663', '5.000000-DIFFUSION EPI 3bTRACEWDFC-52607', '13.000000-T1 3D VIBE AXIAL  avant gado -24651', '12.000000-T2 EG TRA 256 NEW-55362'] --> 14 is t1c, 13 is t1

#ValueError: Only found 4 modalities for exam /home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0025/10-14-2012-NA-RM DE CEREBRO SINCON CONTRASTE-82954
# {'t1c': '6.000000-Obl T1 3D FSPGR IR-40131', 't2': '3.000000-Ax FRFSE T2-30196', 'diffusion': '5.000000-AX DIFUSION B 1500-89195', 'flair': '2.000000-Ax T2 FLAIR ASSET-88274'}
# ['2.000000-Obl T1 3D FSPGR IR IV-86944', '3.000000-Ax FRFSE T2-30196', '5.000000-AX DIFUSION B 1500-89195', '2.000000-Ax T2 FLAIR ASSET-88274', '6.000000-Obl T1 3D FSPGR IR-40131'] --> 2 is t1c, 6 is t1


def rhuh_sort_func(exam_dir):
    date = os.path.basename(exam_dir).replace("-","")
    date = date[4:8] + date[0:2] + date[2:4]
    return date


def find_modalities(exam_dir):
    sequence_dirs = os.listdir(exam_dir)
    modalities = {}

    for d in sequence_dirs:
        d_lower = d.lower().replace("-", " ")
        if "t1 se" in d_lower or "t1 fse" in d_lower or "92851" in d_lower or "24651" in d_lower or "40131" in d_lower:
            modalities["t1"] = os.path.join(exam_dir, d)
            continue
        if "flair" in d.lower():
            modalities["flair"] = os.path.join(exam_dir, d)
            continue
        if "difusion" in d.lower() or "diffusion" in d.lower() or "dwi" in d.lower():
            modalities["diffusion"] = os.path.join(exam_dir, d)
            continue
        if "t2" in d.lower(): #and "se" in d.lower():
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
    # python rhuh_gbm_parser.py -patient_dir /home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM -gpu_device 2
    parser = argparse.ArgumentParser()
    parser.add_argument("-patient_dir", type=str, help="Path to directory containing the patients with exams and DICOM data.")
    parser.add_argument("-gpu_device", type=str, help="GPU id to run on.")
    args = parser.parse_args()
    
    preop_exams = rhuh_parse_exams(args.patient_dir, preop=True)
    print(f"Found {len(preop_exams)} pre-op exams.")
    
    #postop_exams = rhuh_parse_exams(args.patient_dir, preop=False)
    #print(f"Found {len(postop_exams)} post-op exams.")

    """
    for e in preop_exams:
        preprocess_dicom(
            t1 = e["t1"],
            t1c = e["t1c"],
            t2 = e["t2"],
            flair = e["flair"],
            gpu_device=args.gpu_device
            )

    for e in postop_exams:
        preprocess_dicom(
            t1 = e["t1"],
            t1c = e["t1c"],
            t2 = e["t2"],
            flair = e["flair"],
            gpu_device=args.gpu_device
            )
    """
