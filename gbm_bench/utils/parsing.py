import os
import glob
import argparse
from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, TypedDict, Union


class ModalityDict(TypedDict):
    """
    Data type that enforces modality names as keys.
    """
    t1: str
    t1c: str
    t2: str
    flair: str
    diffusion: Optional[str] = None


class Patient(NamedTuple):
    """
    Data type for a patient storing MRI exams and sequences.
    """
    patient_id: str
    patient_dir: str
    exams: List[str]
    sequences: List[ModalityDict]
    info: Optional[dict[str, str]] = None

    def __repr__(self):
        return f"\nPatient( \
                \n Patient ID: {self.patient_id} \
                \n Patient dir: {self.patient_dir} \
                \n Exams: {self.exams} \
                \n Sequences: {self.sequences.__repr__()[0:350]}... \
                \n Info: {self.info} \
                \n )"


class BaseDatasetParser(ABC):
    """
    An abstract base class for a parser to search for exams in a medical dataset.
    """
    #NOTE: add_patient method might be a better interface

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._patients : List[Patient] = []

    def get_patients(self) -> List[Patient]:
        if len(self._patients) == 0:
            raise ValueError(f"No patients parsed. Please use the parse method first or ensure it saves its results in self._patients.")
        return self._patients

    @abstractmethod
    def parse(self) -> None:
        pass


class RHUHParser(BaseDatasetParser):
    
    def find_modalities(self, exam_dir):
        sequence_dirs = [d for d in os.listdir(exam_dir) if d != "preprocessing"]
        modalities = {}

        # Flair, Diffusion, T2 are easy to find. Some t1/t1c have to be hand identified due to similar naming.
        for d in sequence_dirs:
            d_lower = d.lower().replace("-", " ").replace(" ", "")
            if "flair" in d.lower() and not "t1" in d.lower():
                modalities["flair"] = os.path.join(exam_dir, d)
                continue
            if any([pattern in d_lower for pattern in ["dwi", "difusion", "diffusion"]]):
                modalities["diffusion"] = os.path.join(exam_dir, d)
                continue
            if "t2" in d.lower():
                modalities["t2"] = os.path.join(exam_dir, d)
                continue
            if any([pattern in d_lower for pattern in ["t1wprosetgado", "t1fsec"]]):
                modalities["t1c"] = os.path.join(exam_dir, d)
                continue
            if any([pattern in d_lower for pattern in ["92851", "24651", "40131", "40589", "t1se", "t1fse", "t1wse", "t1flair", "t1sagse"]]):
                modalities["t1"] = os.path.join(exam_dir, d)
                continue
            if "t1" in d_lower or "3d" in d_lower:
                modalities["t1c"] = os.path.join(exam_dir, d)

        if len(modalities) != 5:
            raise ValueError(f"Only found {len(modalities)} modalities for exam {exam_dir} \n {modalities} \n {os.listdir(exam_dir)}")

        return ModalityDict(**modalities)

    def parse(self):
        print(f"Parsing {self.root_dir}...")
        patient_dirs = glob.glob(os.path.join(self.root_dir, "RHUH-*"))

        sort_func = lambda x: os.path.basename(x).replace("-", "")[4:8] + os.path.basename(x).replace("-", "")[0:2] + os.path.basename(x).replace("-", "")[2:4]

        for pdir in patient_dirs:
            pid = os.path.basename(pdir)
            patient_exams = glob.glob(os.path.join(pdir, "*-NA-*"))
            
            patient_exams.sort(key = sort_func)

            #patient_exams.sort(key=RHUHParser.sort_date)
            #patient_exams.sort(key=self.sort_date)

            sequences = [self.find_modalities(pexam) for pexam in patient_exams]

            patient = Patient(
                    patient_id=pid,
                    patient_dir=pdir,
                    exams=patient_exams,
                    sequences=sequences,
                    )
            self._patients.append(patient)
        
        print(f"Parsing finished. Found {len(self._patients)} patients.")


if __name__=="__main__":
    # Example
    # python gbm_bench/utils/parsing.py -rhuh_root /home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM
    parser = argparse.ArgumentParser()
    parser.add_argument("-rhuh_root", type=str, help="Path to the directory in the RHUH dataset containing the patient folders.")
    args = parser.parse_args()

    rhuh_parser = RHUHParser(args.rhuh_root)
    rhuh_parser.parse()
    patients = rhuh_parser.get_patients()
    print(f"Found {len(patients)} patients: {patients[0]}...")
    #print(f"{patients[0].sequences}")
