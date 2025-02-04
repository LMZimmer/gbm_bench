# use with a recent release of dcm2niix, e.g. https://github.com/rordenlab/dcm2niix/releases/tag/v1.0.20220720
import argparse
import datetime
import os
import shlex
import subprocess

from auxiliary.turbopath.turbopath import turbopath
from path import Path


def niftiConvert(inputDir, exportDir, fileName):
    try:
        print("*** start ***")

        os.makedirs(exportDir, exist_ok=True)

        # create dcm2niix call
        #dcm2niixargs = (
        #    " -d 9 -f %p -z y -o"  # extra metadata args
        #)
        ## dcm2niixargs = " -f %f -z y -o "
        ## dcm2niixargs = "-d 9 -z y -o"

        readableCmd = (
            # TODO adjust location
            "/home/home/lucas/bin/dcm2niix -d 9 -f "
            + fileName
            + " -z y -o"
            + ' "'
            + exportDir
            + '" "'
            + inputDir
            + '"'
        )

        print(readableCmd)
        command = shlex.split(readableCmd)

        logFilePath = os.path.join(
            exportDir, os.path.basename(exportDir) + "_conversion.log"
        )
        with open(logFilePath, "w") as outFile:
            subprocess.run(command, stdout=outFile, stderr=outFile)

    except Exception as e:
        print("error: " + str(e))
        print("conversion error for:", inputDir)

    time = str(datetime.datetime.now().time())

    print("** finished:", exportDir, "at:", time)


if __name__ == "__main__":
    # python dicom_to_nifti.py -inputDir /home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0001/01-25-2015-NA-RM\ CEREBRAL6NEURNAV-21029/12.000000-Ax\ T1\ 3d\ NEURONAVEGADOR-55128 -exportDir /home/home/lucas/scripts/test/raw -fileName t1c_raw

    parser = argparse.ArgumentParser()
    parser.add_argument("-inputDir", type=str, help="Path to directory containing DICOMs.")
    parser.add_argument("-exportDir", type=str, help="Desired directory for output.")
    parser.add_argument("-fileName", type=str, help="Desired output file name.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]="2"

    niftiConvert(
        inputDir=args.inputDir,
        exportDir=args.exportDir,
        fileName=args.fileName,
    )
