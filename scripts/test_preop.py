from gbm_bench.preprocessing.preprocess import preprocess_dicom


if __name__=="__main__":
    # Example:
    # python scripts/test_preop.py
    dcm2niix_location = "/home/home/lucas/bin/dcm2niix"
    atlas_t1_dir = "/home/home/lucas/bin/miniconda3/envs/brainles/lib/python3.10/site-packages/brainles_preprocessing/registration/atlas/t1_skullstripped_brats_space.nii"
    atlas_tissues_dir = "/home/home/lucas/data/ATLAS/SRI-24/tissues.nii"

    preprocess_dicom(
            t1="test_data/exam1/t1",
            t1c="test_data/exam1/t1c",
            t2="test_data/exam1/t2",
            flair="test_data/exam1/flair",
            dcm2niix_location="/home/home/lucas/bin/dcm2niix",
            atlas_t1_dir=atlas_t1_dir,
            atlas_tissues_dir=atlas_tissues_dir,
            pre_treatment=True,
            cuda_device="2"
            )
