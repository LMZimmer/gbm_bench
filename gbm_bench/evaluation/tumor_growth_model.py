import os
import sys
import shutil
from typing import Dict, List, Optional, Union

from gbm_bench.evaluation.docker import *

from brats.utils.data_handling import input_sanity_check


GROWTH_PRED_NAME_SCHEMA="{subject_id}.nii.gz"


def remove_tmp_folder(folder: Path):
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


def input_sanity_check(
    t1n: Optional[Path | str] = None,
    t1c: Optional[Path | str] = None,
    t2f: Optional[Path | str] = None,
    t2w: Optional[Path | str] = None,
    mask: Optional[Path | str] = None,
):
    """
    Check if input images have the default shape (240, 240, 155) and log a warning if not.
    Supports different input combinations for segmentation and inpainting tasks.

    Args:
        t1n (Path | str): T1n image path (required for segmentation and inpainting)
        t1c (Path | str, optional): T1c image path (required for segmentation)
        t2f (Path | str, optional): T2f image path (required for segmentation)
        t2w (Path | str, optional): T2w image path (required for segmentation)
        mask (Path | str, optional): Mask image path (required for inpainting)
    """

    # Filter out None values to only include provided images
    images = {
        "t1n": t1n,
        "t1c": t1c,
        "t2f": t2f,
        "t2w": t2w,
        "mask": mask,
    }

    # Load and check shapes
    shapes = {
        label: nib.load(img).shape for label, img in images.items() if img is not None
    }

    assert shapes, "No input images provided. At least one image is required."

    if any(shape != (240, 240, 155) for shape in shapes.values()):
        logger.warning(
            "Input images do not have the default shape (240, 240, 155). This might cause issues with some algorithms and could lead to errors."
        )
        logger.warning(f"Image shapes: {shapes}")
        logger.warning(
            "If your data is not preprocessed yet, consider using our preprocessing package: https://github.com/BrainLesion/preprocessing"
        )


@contextmanager
def InferenceSetup(
    log_file: Optional[Path | str] = None,
) -> Generator[Tuple[Path, Path], None, None]:
    """
    Context manager for setting up the inference process. Creates temporary data and output folders and adds a log file handler if requested.

    Yields:
        (data folder, output folder) (Tuple[Path, Path]): Two temporary folders (data folder, output folder)
    """
    tmp_data_folder = Path(tempfile.mkdtemp(prefix="data_"))
    tmp_output_folder = Path(tempfile.mkdtemp(prefix="output_"))

    try:
        yield tmp_data_folder, tmp_output_folder
    finally:
        remove_tmp_folder(tmp_data_folder)
        remove_tmp_folder(tmp_output_folder)


class TumorGrowthModel():
    """A class that utilizes Docker images of tumor growth models to make tumor grwoth predictions."""

    def __init__(self, algorithm: str, cuda_devices: Optional[str] = "0", force_cpu: bool = False):
        pass

    def _standardize_input_files(self, data_folder: str, subject_id: str, inputs: Dict[str, str], subject_modality_separator: str) -> None:
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

        subject_folder = os.path.join(data_folder, )
        subject_folder = data_folder / subject_id
        subject_folder.mkdir(parents=True, exist_ok=True)
        try:
            for modality, path in inputs.items():
                shutil.copy(
                    path,
                    subject_folder
                    / f"{subject_id}{subject_modality_separator}{modality}.nii.gz",
                )
        except FileNotFoundError as e:
            logger.error(f"Error while standardizing files: {e}")
            logger.error(
                "If you use batch processing please ensure the input files are in the correct format, i.e.:\n A/A-t1c.nii.gz, A/A-t1n.nii.gz, A/A-t2f.nii.gz, A/A-t2w.nii.gz"
            )
            sys.exit(1)

        # sanity check inputs
        input_sanity_check(
            t1c=inputs.get("t1c"),
            t1n=inputs.get("t1n"),
            t2f=inputs.get("t2f"),
            t2w=inputs.get("t2w"),
        )

    def predict_single(self, t1: str, gm: str, wm: str, csf: str, tumorseg: str, pet: str, output_file: str, log_file: Optional[str] = None) -> None:
        """Predict tumor growth on a single subject with the provided images and save the result to the output file.

        Args:
            t1c (Path | str): Path to the T1c image
            t1n (Path | str): Path to the T1n image
            t2f (Path | str): Path to the T2f image
            t2w (Path | str): Path to the T2w image
            output_file (Path | str): Path to save the segmentation
            log_file (Path | str, optional): Save logs to this file
        """
        inputs = {"t1": t1, "gm": gm, "wm": wm, "csf": csf, "tumorseg": tumorseg, "pet": pet}


        with InferenceSetup(log_file=log_file) as (tmp_data_folder, tmp_output_folder):

            # the id here is arbitrary
            subject_id = self.algorithm.run_args.input_name_schema.format(id=0)

            self._standardize_single_inputs(
                data_folder=tmp_data_folder,
                subject_id=subject_id,
                inputs=inputs,
                subject_modality_separator=self.algorithm.run_args.subject_modality_separator,
            )

            run_container(
                algorithm=self.algorithm,
                data_path=tmp_data_folder,
                output_path=tmp_output_folder,
                cuda_devices=self.cuda_devices,
                force_cpu=self.force_cpu,
            )
            self._process_single_output(
                tmp_output_folder=tmp_output_folder,
                subject_id=subject_id,
                output_file=output_file,
            )
            logger.info(f"Saved output to: {Path(output_file).absolute()}")

    def _process_single_output(
        self, tmp_output_folder: Path | str, subject_id: str, output_file: Path
    ) -> None:
        """
        Process the output of a single inference run and save it in the specified file.

        Args:
            tmp_output_folder (Path | str): Folder with the algorithm output
            subject_id (str): Subject ID of the output
            output_file (Path): Path to the desired output file
        """
        # rename output
        if self.task == Task.MISSING_MRI:
            # Missing MRI has no fixed names since the missing modality differs and is included in the name
            algorithm_output = Path(tmp_output_folder).iterdir().__next__()
        else:
            algorithm_output = Path(tmp_output_folder) / OUTPUT_NAME_SCHEMA[
                self.task
            ].format(subject_id=subject_id)

        # ensure path exists and rename output to the desired path
        output_file = Path(output_file).absolute()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(algorithm_output, output_file)


if __name __ == "__main__":
    pass
