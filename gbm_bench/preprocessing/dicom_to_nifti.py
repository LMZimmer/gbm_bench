import os
import shlex
import argparse
import subprocess
from typing import List
from pathlib import Path
from loguru import logger
from auxiliary.turbopath.turbopath import turbopath
from gbm_bench.utils.constants import DCM2NIIX_LOCATION


def remove_postfixes(export_dir: Path) -> None:
    """
    Remove postfixes created by dcm2niix (e.g. '_real') from filenames in the given directory.

    Parameters:
        outdir (Path): The directory containing files to be renamed.
    """
    # Iterate over all files in the directory.
    for f in export_dir.iterdir():
        if f.is_file() and ("_" in f.name) and not f.name.endswith(".log"):
            # Split the filename into the base (with possible postfixes) and extension(s)
            parts = f.name.split(".")
            # Extract the modality (before the first underscore)
            modality = parts[0].split("_")[0]
            # Construct the new filename using the modality and the original extensions
            new_name = ".".join([modality] + parts[1:])
            new_path = export_dir / new_name
            f.rename(new_path)
            logger.info(f"Renamed postfix file {file} to {new_path}.")


def niftiConvert(input_dir: Path, export_dir: Path, outfile: str, dcm2niix_location: Path = DCM2NIIX_LOCATION) -> None:
    """
    Convert DICOM files to NIfTI format using dcm2niix.

    Parameters:
        input_dir (Path): Directory containing the raw DICOM files.
        export_dir (Path): Directory where the converted NIfTI files will be saved.
        outfile (Path): The filename template for the output files.
        dcm2niix_location (Path, optional): The location of the dcm2niix executable.
    """
    try:
        export_dir.mkdir(parents=True, exist_ok=True)
        cmd_readable = (
            str(dcm2niix_location)
            + " -d 9 -f "
            + outfile
            + " -z y -o"
            + ' "'
            + str(export_dir)
            + '" "'
            + str(input_dir)
            + '"'
        )

        logger.info(f"Running: {cmd_readable}")
        cmd = shlex.split(cmd_readable)

        log_file = export_dir / f"{export_dir.name}_conversion.log"
        with open(log_file, "w") as logf:
            subprocess.run(cmd, stdout=logf, stderr=logf)

        remove_postfixes(export_dir)
        logger.debug(f"Nifti conversion complete for {input_dir}.")

    except Exception as e:
        logger.error(f"Error while trying to convert {input_dir} via {cmd_readable}: {e}")


if __name__ == "__main__":
    # Example
    # python gbm_bench/preprocessing/dicom_to_nifti.py -dcm2niix_loc /home/home/lucas/bin/dcm2niix
    parser = argparse.ArgumentParser()
    parser.add_argument("-dcm2niix_loc", type=str, help="Path to your dcm2niix executable.")
    args = parser.parse_args()

    niftiConvert(
        input_dir=Path("test_data/exam1/t1c"),
        export_dir=Path("./tmp_test_dcm2nii"),
        outfile="t1c",
        dcm2niix_location=args.dcm2niix_loc
    )
