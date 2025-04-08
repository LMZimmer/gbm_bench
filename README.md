# GBMbench
Glioblastoma Model Benchmark (working title) is a work in progress with the goal of assessing the possible benefit of Glioblastoma models for better radiotherapy planning.


## Features
- Standardized processing pipeline for Brain MRIs
- Extensible benchmark framework for dockered Glioblastoma models
- Easy-to-use minimal API


## Setting up

### Prerequisites
- docker: A Docker installation for running BRATS models as well as Glioblstoma models. Notes on the docker setup with GPU support can be found below.

- dicom2niix: If you plan to process raw DICOM data.

You require a Docker installation for runnign BRATS models as well as Glioblstoma models.

Any remaining dependencies are Python packages, such as brainles-preprocessing, brats and antspy, are listed in requirements.txt. They are installed alongside the Python package explained in the following section.

### Installation

```bash
git clone https://github.com/LMZimmer/gbm_bench.git
cd gbm_bench
pip install .
```

### Docker and NVIDIA Container Toolkit Setup

- **Docker**: Installation instructions on the official [website](https://docs.docker.com/get-docker/)
- **NVIDIA Container Toolkit**: Refer to the [NVIDIA install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and the official [GitHub page](https://github.com/NVIDIA/nvidia-container-toolkit)


## Adding new growth models

This repository can be used to perform inference or benchmark with your own tumor growth model. To this end, you need to create a docker image of your growth model. The following sections serve as guideline on how the image should be created. 

### Directory structure
Data is passed to the docker container by mounting a directory /mlcube\_io0 and the output is read from a directory /mlcube\_io1. The directory structure is assumed as:

INPUT:

```bash
/mlcube\_io0
   ┗ Patient-00000
      ┣ 00000-t1c.nii.gz
      ┣ 00000-gm.nii.gz
      ┣ 00000-wm.nii.gz
      ┣ 00000-csf.nii.gz
      ┣ 00000-tumorseg.nii.gz
      ┗ 00000-pet.nii.gz
```

OUTPUT:

```bash
/mlcube\_io1
   ┗ 00000.nii.gz
```

### Dockerfile
As long as the container adheres to the directory structure outlined above, there are no further requirements to the container. An example Dockerfile could be:

```bash
FROM <python:3.8-slim>
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code to workdir
COPY . .
ENTRYPOINT ["python", "inference.py"]
```
