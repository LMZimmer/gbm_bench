# GBMbench
Glioblastoma Model Benchmark (working title) is a work in progress with the goal of assessing the possible benefit of Glioblastoma models for better radiotherapy planning.


## Features
- Standardized processing pipeline for Brain MRIs
- Extensible benchmark framework for dockered Glioblastoma models
- Easy-to-use minimal API


## Setting up

### Prerequisites
- docker: A Docker installation for running BRATS models as well as Glioblstoma models. Notes on the docker setup with GPU support can be found in [Docker and NVIDIA Container Toolkit Setup][### Docker and NVIDIA Container Toolkit Setup]. Instructions on adding custom models are given in [Adding new growth models][## Adding new growth models].

- dicom2niix: If you plan to process raw DICOM data.

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
/mlcube_io0
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
/mlcube_io1
   ┗ 00000.nii.gz
```

### Dockerfile
As long as the container adheres to the directory structure outlined above, there are no further requirements to the container. An example Dockerfile could be:

```bash
# Image and environment variables
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code to workdir
COPY . .
ENTRYPOINT ["python3", "inference.py"]
```


## Adding new datasets

Datasets are handled with the LongitudinalDataset class in utils.parsing. This class can parse datasets that have a specific directory structure and can save/load the paths as json. Therefore, the prefered method would be to create a json file that can be read from this class. An example for this is given e.g. in data/datasets/rhuh.json. LongitudinalDataset can also automatically parse specific directory structures, but then identifying preop, postop, followup is an issue.
