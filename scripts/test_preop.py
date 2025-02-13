from gbm_bench.preprocessing.preprocess import preprocess_dicom


if __name__=="__main__":
    # Example:
    # python scripts/test_preop.py
    preprocess_dicom(
            t1="test_data/exam1/t1",
            t1c="test_data/exam1/t1c",
            t2="test_data/exam1/t2",
            flair="test_data/exam1/flair",
            pre_treatment=True,
            gpu_device="2"
            )
