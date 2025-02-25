import os
import argparse
from gbm_bench.utils.utils import merge_pdfs
from gbm_bench.utils.parsing import RHUHParser
from gbm_bench.utils.visualization import plot_mri_with_segmentation, plot_model_multislice


if __name__ == "__main__":
    # Example:
    # python scripts/visualize_rhuh.py

    os.environ["CUDA_VISIBLE_DEVICES"]="4"

    rhuh_root = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
    rhuh_parser = RHUHParser(root_dir=rhuh_root)
    rhuh_parser.parse()
    patients = rhuh_parser.get_patients()

    outfiles = []
    for ind, patient in enumerate(patients):
        print(f"Creating plots {ind}/{len(patients)}...")
        patient_identifier = patient["patient_id"]
        exam_identifier = patient["exam_ids"][0]     # First exam is pre-op
        algorithm_identifier = "lmi"
        preprocessing_dir = os.path.join(patient["exams"][0], "preprocessing")
        os.makedirs("./tmp", exist_ok=True)
        outfile = f"./tmp/{patient_identifier}_lmi.pdf"
        outfiles.append(outfile)
        
        #plot_mri_with_segmentation(
        #        patient_identifier=patient_identifier,
        #        exam_identifier=exam_identifier,
        #        algorithm_identifier=algorithm_identifier,
        #        preprocessing_dir=preprocessing_dir,
        #        outfile=outfile
        #        )

        plot_model_multislice(
                patient_identifier=patient_identifier,
                exam_identifier=exam_identifier,
                algorithm_identifier=algorithm_identifier,
                preprocessing_dir=preprocessing_dir,
                outfile=outfile
                )

    # Merge all PDFs for this algorithm into one for the patient
    combined_pdf_path = f"./tmp/RHUH_{algorithm_identifier}.pdf"
    outfiles.sort()
    merge_pdfs(outfiles, combined_pdf_path)

    # Delete single files
    for f in outfiles:
        os.remove(f)
