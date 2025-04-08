import os
import time
import docker
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Tuple
from docker.errors import DockerException


def _is_cuda_available() -> bool:
    """Check if CUDA is available on the system by trying to run nvidia-smi."""
    try:
        # Attempt to run `nvidia-smi` to check for CUDA.
        # This command should run successfully if NVIDIA drivers are installed and GPUs are present.
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except:
        return False


def _handle_device_requests(cuda_devices: str, force_cpu: bool) -> List[docker.types.DeviceRequest]:
    """Handle the device requests for the docker container (request cuda or cpu).

    Args:
        cuda_devices (str): The CUDA devices to use
        force_cpu (bool): Whether to force CPU execution
    """
    cuda_available = _is_cuda_available()
    if not cuda_available or force_cpu:
        # empty device requests => run on CPU
        logger.info("Forcing CPU execution")
        return []
    # request gpu with chosen devices
    return [
        docker.types.DeviceRequest(device_ids=[cuda_devices], capabilities=[["gpu"]])
    ]


def _get_volume_mappings(data_path: str, output_path: str) -> Dict:
    """Get the volume mappings for the docker container.

    Args:
        data_path: The path to the input data
        output_path: The path to save the output

    Returns:
        Dict: The volume mappings
    """
    # TODO: add support for recommended "ro" mount mode for input data
    # data = mlcube_io0, output = mlcube_io1
    return {
        volume.absolute(): {
            "bind": f"/mlcube_io{i}",
            "mode": "rw",
        }
        for i, volume in enumerate(
            [data_path, , output_path]
        )
    }


def _ensure_image(algorithm: str, model_path: str) -> str:
    """
    Checks if algorithm:latest image is present. If not loads model_path into docker. Returns the image tag.

    Args:
        algorithm: Algorithm name
        model_path: Path to the growth model docker image
    """
    image_tag = f"{algorithm}:latest"

    try:
        # Check if the docker image exists by trying to inspect it
        subprocess.run(
            ["docker", "inspect", image_tag],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Image '{image_tag}' found. Skipping loading the image.")
    
    except subprocess.CalledProcessError:
        logger.info(f"Image '{image_tag}' not found. Loading image from '{model_path}'...")
        try:
            # Load the docker image
            subprocess.run(
                ["docker", "load", "-i", model_path],
                check=True
            )
            logger.info(f"Image '{image_tag}' loaded successfully from '{model_path}'.")
        except subprocess.CalledProcessError as e:
            raise e #TODO: meaningful exception
    
    return image_tag


def _observe_docker_output(container: docker.models.containers.Container) -> str:
    """Observe the output of a running docker container and display a spinner. On Errors log container output.

    Args:
        container (docker.models.containers.Container): The container to observe
    """
    # capture the output
    container_output = container.attach(
        stdout=True, stderr=True, stream=True, logs=True
    )

    # Display spinner while the container is running
    with Console().status("Running inference..."):
        # Wait for the container to finish
        exit_code = container.wait()
        container_output = "\n\r".join(
            [line.decode("utf-8", errors="replace") for line in container_output]
        )
        # Check if the container exited with an error
        if exit_code["StatusCode"] != 0:
            logger.error(f">> {container_output}")
            raise RuntimeError(
                "Container finished with an error. See logs above for details."
            )

    return container_output


def _sanity_check_output(data_path: str, output_path: str, container_output: str) -> None:
    """Sanity check that the number of output files matches the number of input files and the output is not empty.

    Args:
        data_path: The path to the input data
        output_path: The path to the output data
        container_output: The output of the docker container
    """
    outputs = list(output_path.iterdir())
    if len(outputs) != 1:
        logger.error(f"Docker container output: \n\r{container_output}")
        raise RuntimeError(f"Expected 1 output file but got {len(outputs)}. Please check the logging output of the docker container for more information.")


def run_container(algorithm: str, model_path: str, data_path: str, output_path: str, cuda_devices: str, force_cpu: bool) -> None:
    """Run a docker container for the provided algorithm.

    Args:
        algorithm: Name of the algorithm
        model_path: Path to the growth model docker image.
        data_path: The path to the input data
        output_path: The path to save the output
        cuda_devices: The CUDA devices to use
        force_cpu: Whether to force CPU execution
        internal_external_name_map: Dictionary mapping internal name (in standardized format) to external subject name provided by user (only used for batch inference)
    """
    # ensure output folder exists
    output_path.mkdir(parents=True, exist_ok=True)

    volume_mappings = _get_volume_mappings(
        data_path=data_path,
        output_path=output_path
    )
    logger.debug(f"Volume mappings: {volume_mappings}")

    # device setup
    device_requests = _handle_device_requests(cuda_devices=cuda_devices, force_cpu=force_cpu)
    logger.debug(f"GPU Device requests: {device_requests}")

    # load image if necessary
    image_tag = _ensure_image(algorithm, model_path)

    # Run the container
    logger.info(f"{'Starting growth prediction'}")
    start_time = time.time()
    container = client.containers.run(
        image=image_tag,
        volumes=volume_mappings,
        device_requests=device_requests,
        network_mode="none",
        detach=True,
        remove=True,
        shm_size="20gb", #TODO
        #user=f"{os.getuid()}:{os.getgid()}"  #this line disables running as root
    )
    container_output = _observe_docker_output(container=container)
    _sanity_check_output(
        data_path=data_path,
        output_path=output_path,
        container_output=container_output
    )

    logger.debug(f"Docker container output: \n\r{container_output}")

    logger.info(f"Finished inference in {time.time() - start_time:.2f} seconds")
