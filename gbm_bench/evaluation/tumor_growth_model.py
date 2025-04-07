import os
import sys
import shutil
from typing import Dict, List, Optional, Union

from gbm_bench.evaluation.docker import *


GROWTH_PRED_SCHEMA="{subject_id}.nii.gz"


def remove_tmp_folder(folder: str):
    """Remove a temporary folder and log a warning if it fails.

    Args:
        folder (Path): Path to the folder to be removed
    """
    try:
        shutil.rmtree(folder)
    except PermissionError as e:
        logger.warning(
            f"Failed to remove temporary folder {folder}. This is most likely caused by bad permission management of the docker container. \nError: {e}"
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to delete folder {folder}. {e}")


@contextmanager
def InferenceSetup(log_file: str = None) -> Generator[Tuple[Path, Path], None, None]:
    """
    Context manager for setting up the inference process. Creates temporary data and output folders and adds a log file handler if requested.

    Yields:
        (data folder, output folder) (Tuple[Path, Path]): Two temporary folders (data folder, output folder)
    """
    if log_file is not None:
        logger_id = add_log_file_handler(log_file)

    tmp_data_folder = Path(tempfile.mkdtemp(prefix="data_"))
    tmp_output_folder = Path(tempfile.mkdtemp(prefix="output_"))

    try:
        yield tmp_data_folder, tmp_output_folder
    finally:
        remove_tmp_folder(tmp_data_folder)
        remove_tmp_folder(tmp_output_folder)

        if log_file is not None:
            logger.remove(logger_id)


class TumorGrowthModel():
    """A class that utilizes Docker images of tumor growth models to make tumor grwoth predictions."""

    def __init__(self, algorithm: str, cuda_devices: Optional[str] = "0", force_cpu: bool = False):
        self.algorithm = algorithm
        self.cuda_devices = cuda_devices
        self.force_gpu = force_cpu

        self.algorithm_list = load_algorithms(file_path=algorithms_file_path) #TODO: This function, list algorithms function, and check if something has to be changed in docker.py

    def _standardize_input_files(self, data_folder: str, subject_id: int, inputs: Dict[str, str]) -> None:
        """Standardize the input images for a single subject to match requirements of all algorithms and save them in @data_folder/@subject_id.
        Example:
                Patient-00000-000 \n
                ┣ 00000-000-t1c.nii.gz \n
                ┣ 00000-000-gm.nii.gz \n
                ┣ 00000-000-wm.nii.gz \n
                ┣ 00000-000-csf.nii.gz \n
                ┣ 00000-000-tumorseg.nii.gz \n
                ┗ 00000-000-pet.nii.gz \n

        Args:
            data_folder: Temporary folder to cache patient images
            subject_id: Subject ID to be used for the folder and filenames
            inputs: Dictionary with the input images
            subject_modality_separator: Separator between the subject ID and the modality
        """

        subject_folder = os.path.join(data_folder, f"Patient-{subject_id}")
        subject_folder.mkdir(parents=True, exist_ok=True)
        try:
            for modality, path in inputs.items():
                shutil.copy(
                        path,
                        os.path.join(subject_folder, f"{subject_id}-{modality}.nii.gz")
                        )
        except FileNotFoundError as e:
            logger.error(f"Error while standardizing files: {e}")
            sys.exit(1)

    def _process_output(
        self, tmp_output_folder: str, subject_id: str, output_file: str) -> None:
        """
        Process the output of a docker growth model and save it in the specified file.

        Args:
            tmp_output_folder: Folder with the algorithm output
            subject_id: Subject ID of the output
            output_file: Path to the desired output file
        """
        # rename output
        algorithm_output = os.path.join(tmp_output_folder, GROWTH_PRED_SCHEMA.format(subject_id=subject_id))

        # ensure path exists and rename output to the desired path
        output_file = os.path.abspath(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        shutil.move(algorithm_output, output_file)

    def predict_single(self, t1c: str, gm: str, wm: str, csf: str, tumorseg: str, pet: str, output_file: str, log_file: Optional[str] = None) -> None:
        """Predict tumor growth on a single subject with the provided images and save the result to the output file.

        Args:
            t1c: Path to t1c image
            gm: Path to gm probability map
            wm: Path to wm probability map
            csf: Path to csf probability map
            tumorseg: Path to tumor segmentation
            pet: Path to pet image
            output_file: Path to save the segmentation
            log_file: Save logs to this file
        """
        inputs = {"t1c": t1, "gm": gm, "wm": wm, "csf": csf, "tumorseg": tumorseg, "pet": pet}


        with InferenceSetup(log_file=log_file) as (tmp_data_folder, tmp_output_folder):

            # the id here is arbitrary
            subject_id = "00000"

            self._standardize_input_files(
                data_folder=tmp_data_folder,
                subject_id=subject_id,
                inputs=inputs
            )

            run_container(
                algorithm=self.algorithm,
                data_path=tmp_data_folder,
                output_path=tmp_output_folder,
                cuda_devices=self.cuda_devices,
                force_cpu=self.force_cpu,
            )

            self._process_output(
                tmp_output_folder=tmp_output_folder,
                subject_id=subject_id,
                output_file=output_file,
            )
            logger.info(f"Saved output to: {os.path.abspath(output_file)}")


if __name __ == "__main__":
    pass
