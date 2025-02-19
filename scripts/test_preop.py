from gbm_bench.preprocessing.preprocess import preprocess_dicom


if __name__=="__main__":
    # Example:
    # python scripts/test_preop.py
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="1", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    dcm2niix_location = "/home/home/lucas/bin/dcm2niix"

    preprocess_dicom(
            t1="test_data/exam1/t1",
            t1c="test_data/exam1/t1c",
            t2="test_data/exam1/t2",
            flair="test_data/exam1/flair",
            dcm2niix_location="/home/home/lucas/bin/dcm2niix",
            pre_treatment=True,
            cuda_device="2"
            )
