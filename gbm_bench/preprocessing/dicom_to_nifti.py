import os
import shlex
import argparse
import subprocess
from typing import List
from auxiliary.turbopath.turbopath import turbopath


def remove_postfixes(outdir: str) -> None:
    # removes postfixes created by dcm2niix (e.g. _real)
    postfix_files = [f for f in os.listdir(outdir) if "_" in f and not f.endswith(".log")]
    for pf in postfix_files:
        modality, file_extensions = pf.split(".")[0], pf.split(".")[1:]
        modality = modality.split("_")[0]
        nf = ".".join([modality] + file_extensions)
        old, new = os.path.join(outdir, pf), os.path.join(outdir, nf)
        os.rename(old, new)
        print(f"Renamed postfix file {old} to {new}.")


def niftiConvert(input_dir: str, export_dir: str, outfile: str, dcm2niix_location: str) -> None:
    try:
        os.makedirs(export_dir, exist_ok=True)
        cmd_readable = (
            dcm2niix_location
            + " -d 9 -f "
            + outfile
            + " -z y -o"
            + ' "'
            + export_dir
            + '" "'
            + input_dir
            + '"'
        )

        print(cmd_readable)
        cmd = shlex.split(cmd_readable)

        log_file = os.path.join(
            export_dir, os.path.basename(export_dir) + "_conversion.log"
        )
        with open(log_file, "w") as outFile:
            subprocess.run(cmd, stdout=outFile, stderr=outFile)

        remove_postfixes(export_dir)

    except Exception as e:
        print("error: " + str(e))
        print("conversion error for:", input_dir)


if __name__ == "__main__":
    # Example
    # python gbm_bench/preprocessing/dicom_to_nifti.py -dcm2niix_loc /home/home/lucas/bin/dcm2niix

    parser = argparse.ArgumentParser()
    parser.add_argument("-dcm2niix_loc", type=str, help="Path to your dcm2niix executable.")
    args = parser.parse_args()

    niftiConvert(
        input_dir="test_data/exam1/t1c",
        export_dir="./tmp_test_dcm2nii",
        outfile="t1c",
        dcm2niix_location=args.dcm2niix_loc
    )
