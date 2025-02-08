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

#ValueError: Only found 4 modalities for exam /home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0029/10-08-2013-NA-RM CEREBRO-16983 
# {'t2': '/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0029/10-08-2013-NA-RM CEREBRO-16983/6.000000-Ax T2 FRFSE-45087', 'flair': '/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0029/10-08-2013-NA-RM CEREBRO-16983/3.000000-Sag T1 Flair-58836', 'diffusion': '/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0029/10-08-2013-NA-RM CEREBRO-16983/5.000000-DIFUSION 2000b-92733', 't1c': '/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0029/10-08-2013-NA-RM CEREBRO-16983/11.000000-SAG T1 3D C-32615'} 
# ['6.000000-Ax T2 FRFSE-45087', '4.000000-Ax T2 FLAIR-41390', '5.000000-DIFUSION 2000b-92733', '3.000000-Sag T1 Flair-58836', '11.000000-SAG T1 3D C-32615'] --> t1 flair

#ValueError: Only found 4 modalities for exam /home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0006/11-19-2017-NA-RM CEREBRAL-03097 
# {'t2': '/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0006/11-19-2017-NA-RM CEREBRAL-03097/8.000000-Ax T2 FSE Prop.-81294', 't1': '/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0006/11-19-2017-NA-RM CEREBRAL-03097/10.000000-Ax T1 FSE C-28409', 'diffusion': '/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0006/11-19-2017-NA-RM CEREBRAL-03097/4.000000-Ax Difusion 1000-2000b-22482', 'flair': '/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0006/11-19-2017-NA-RM CEREBRAL-03097/3.000000-Ax T2 FLAIR-03821'} 
# ['8.000000-Ax T2 FSE Prop.-81294', '6.000000-Ax T1 FSE-40589', '10.000000-Ax T1 FSE C-28409', '4.000000-Ax Difusion 1000-2000b-22482', '3.000000-Ax T2 FLAIR-03821'] --> this t1 and t1c only differ by a c..., could add a check that one modality cant be picked more than once but its a nice check 


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
    # python rhuh_gbm_parser.py -patient_dir /home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM -gpu_device 2
    parser = argparse.ArgumentParser()
    parser.add_argument("-patient_dir", type=str, help="Path to directory containing the patients with exams and DICOM data.")
    parser.add_argument("-gpu_device", type=str, help="GPU id to run on.")
    args = parser.parse_args()
    
    preop_exams = rhuh_parse_exams(args.patient_dir, preop=True)
    print(f"Found {len(preop_exams)} pre-op exams: {preop_exams[0]}...")
    
    #postop_exams = rhuh_parse_exams(args.patient_dir, preop=False)
    #print(f"Found {len(postop_exams)} post-op exams: {postop_exams[0]}...")

    for ind, e in enumerate(preop_exams[1:2]):
        print(f"Processing exam {ind}: {e} \n")
        preprocess_dicom(
            t1 = e["t1"],
            t1c = e["t1c"],
            t2 = e["t2"],
            flair = e["flair"],
            gpu_device=args.gpu_device
            )
    
    """
    for e in postop_exams:
        preprocess_dicom(
            t1 = e["t1"],
            t1c = e["t1c"],
            t2 = e["t2"],
            flair = e["flair"],
            gpu_device=args.gpu_device
            )
    """
