import argparse
import datetime
import os
import shlex
import subprocess
from auxiliary.turbopath.turbopath import turbopath
from path import Path


def remove_postfixes(outdir):
    # removes postfixes created by dcm2niix (e.g. _real)
    postfix_files = [f for f in os.listdir(outdir) if "_" in f and not f.endswith(".log")]
    for pf in postfix_files:
        modality, file_extensions = pf.split(".")[0], pf.split(".")[1:]
        modality = modality.split("_")[0]
        nf = ".".join([modality] + file_extensions)
        old, new = os.path.join(outdir, pf), os.path.join(outdir, nf)
        os.rename(old, new)
        print(f"Renamed postfix file {old} to {new}.")


def niftiConvert(input_dir, export_dir, outfile, dcm2niix_location):
    try:
        os.makedirs(export_dir, exist_ok=True)
        cmd_readable = (
            #"/home/home/lucas/bin/dcm2niix -d 9 -f "
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
    # python gbm_bench/preprocessing/dicom_to_nifti.py -input_dir test_data/exam1/t1c -export_dir tmp_test_dcm2nii -outfile t1c -dcm2niix_loc /home/home/lucas/bin/dcm2niix

    parser = argparse.ArgumentParser()
    parser.add_argument("-input_dir", type=str, help="Path to directory containing DICOMs.")
    parser.add_argument("-export_dir", type=str, help="Desired directory for output.")
    parser.add_argument("-outfile", type=str, help="Desired output file name.")
    parser.add_argument("-dcm2niix_loc", type=str, help="Path to your dcm2niix executable.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]="2"

    niftiConvert(
        input_dir=args.input_dir,
        export_dir=args.export_dir,
        outfile=args.outfile,
        dcm2niix_location=args.dcm2niix_loc
    )
