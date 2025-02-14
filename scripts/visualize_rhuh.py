import os
import argparse
from gbm_bench.utils.parsing import RHUHParser


if __name__ == "__main__":
    # Example:
    # python scipts/visualize_rhuh.py

    rhuh_root = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
    rhuh_parser = RHUHParser(root_dir=rhuh_root)
    rhuh_parser.parse()
    patients = rhuh_parser.get_patients()

    clear_old_visualization = False

    #TODO:
    patient_exams = []
    for exam in preop_exams:
        print(f"{exam}")
        patient_identifier = exam.split("/")[-2]
        exam_identifier = "0"
        algorithm_identifier = "BRATS"
        output_pdf = os.path.join(exam, f"preprocessing/visualization/{algorithm_identifier}_{patient_identifier}_{exam_identifier}.pdf")
        
        if clear_old_visualization:
            print(f"Clearing old visualizations in {os.path.dirname(output_pdf)}")
            shutil.rmtree(os.path.dirname(output_pdf))
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

        plot_exam(
                patient_identifier=patient_identifier,
                exam_identifier=exam_identifier,
                algorithm_identifier=algorithm_identifier,
                exam_path=exam,
                output_pdf=output_pdf,
                )
        patient_exams.append(output_pdf)

        # Merge all PDFs for this algorithm into one for the patient
        #combined_pdf_path = f"{patient_report_folder}/combined/{algorithm}_{patient.name}_combined.pdf"
        #merge_pdfs(patient_exams, combined_pdf_path)


    # THIS WAS LMI VISUALIZATION
    os.environ["CUDA_VISIBLE_DEVICES"]="7"

    # Loop over data and algorithms
    data_folder = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
    preop_exams = rhuh_parse_exams(data_folder, preop=True)

    patient_exams = []
    for exam in preop_exams:
        print(f"{exam}")
        patient_identifier = exam.split("/")[-2]
        exam_identifier = "0"
        algorithm_identifier = "LMI"
        output_pdf = os.path.join(exam, f"preprocessing/visualization/{algorithm_identifier}_{patient_identifier}_{exam_identifier}.pdf")
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

        plot_exam(
                patient_identifier=patient_identifier,
                exam_identifier=exam_identifier,
                algorithm_identifier=algorithm_identifier,
                exam_path=exam,
                output_pdf=output_pdf,
                )
        patient_exams.append(output_pdf)

        # Merge all PDFs for this algorithm into one for the patient
        #combined_pdf_path = f"{patient_report_folder}/combined/{algorithm}_{patient.name}_combined.pdf"
        #merge_pdfs(patient_exams, combined_pdf_path)
