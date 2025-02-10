import os
from preprocessing.dicom_to_nifti import niftiConvert
from preprocessing.brainles_preprocessing import run_preprocessing
from preprocessing.tumor_segmentation import run_brats
from preprocessing.tissue_segmentation import generate_healthy_brain_mask, run_tissue_seg_registration


def preprocess_dicom(t1, t1c, t2, flair, pre_treatment=True, gpu_device="2"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

    # Step 1: DICOM to NifTi conversion
    print("Converting DICOM to NifTi...")
    dicom_modalities = {
            "t1" : t1,
            "t1c" : t1c,
            "t2" : t2,
            "flair" : flair
            }

    for modality_name, dicom_folder in dicom_modalities.items():
        out_dir = os.path.join(os.path.dirname(dicom_folder), "preprocessing")
        nifti_dir = os.path.join(out_dir, "nifti_conversion")

        #niftiConvert(
        #        inputDir=dicom_folder,
        #        exportDir=nifti_dir,
        #        fileName=modality_name
        #        )

        print("NifTi conversion complete.")

    # Step 2: Normalize, co-registrate, skull strip
    print("Running normalization, co-registration and skull stripping...")
    preprocessed_dir = os.path.join(out_dir, "skull_stripped")

    #run_preprocessing(
    #        t1=os.path.join(nifti_dir, "t1.nii.gz"),
    #        t1c=os.path.join(nifti_dir, "t1c.nii.gz"),
    #        t2=os.path.join(nifti_dir, "t2.nii.gz"),
    #        flair=os.path.join(nifti_dir, "flair.nii.gz"),
    #        outdir=preprocessed_dir
    #        )

    print("Skull stripping complete.")

    # Step 3: Segment tumor
    print("Segmenting tumor...")
    tumor_outdir = os.path.join(out_dir, "tumor_segmentation")
    tumor_outfile = os.path.join(tumor_outdir, "tumor_seg.nii.gz")
    os.makedirs(tumor_outdir, exist_ok=True)

    #run_brats(
    #        t1=os.path.join(preprocessed_dir, "t1_bet_normalized.nii.gz"),
    #        t1c=os.path.join(preprocessed_dir, "t1c_bet_normalized.nii.gz"),
    #        t2=os.path.join(preprocessed_dir, "t2_bet_normalized.nii.gz"),
    #        flair=os.path.join(preprocessed_dir, "flair_bet_normalized.nii.gz"),
    #        outfile=tumor_outfile,
    #        pre_treatment=pre_treatment,
    #        cuda_device="2"
    #        )

    print("Tumor segmentation complete.")

    # Step 4: Segment tissue
    print("Segmenting tissue...")
    brain_mask_dir = os.path.join(preprocessed_dir, "t1c_bet_mask.nii.gz")
    healthy_mask_dir = os.path.join(tumor_outdir, "healthy_brain_mask.nii.gz")

    generate_healthy_brain_mask(
            brain_mask_file=brain_mask_dir,
            tumor_mask_file=tumor_outfile,
            outdir=healthy_mask_dir)

    tissue_seg_dir = os.path.join(out_dir, "tissue_segmentation")
    run_tissue_seg_registration(
            t1_file=os.path.join(preprocessed_dir, "t1c_bet_normalized.nii.gz"),
            outdir=tissue_seg_dir,
            healthy_mask_dir=healthy_mask_dir,
            brain_mask_dir=brain_mask_dir,
            refit_brain=True)

    print("Tissue segmentation complete.")


if __name__ == "__main__":
    # Example:
    # python preprocess.py

    # Pre-treatment example
    preprocess_dicom(
            t1="test_data/exam1/t1",
            t1c="test_data/exam1/t1c",
            t2="test_data/exam1/t2",
            flair="test_data/exam1/flair",
            pre_treatment=True,
            gpu_device="4"
            )
    
    # Post-treatment example
    #preprocess_dicom(
    #        t1="test_data/exam2/t1",
    #        t1c="test_data/exam2/t1c",
    #        t2="test_data/exam2/t2",
    #        flair="test_data/exam2/flair",
    #        pre_treatment=False,
    #        gpu_device="2"
    #        )
