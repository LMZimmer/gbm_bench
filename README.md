# GBMbench
Glioblastoma Model Benchmark (working title) is a work in progress with the goal of assessing the possible benefit of Glioblastoma models for better radiotherapy planning.


## Prerequisites

- docker: A Docker installation for running BRATS models as well as Glioblstoma models. Notes on the docker setup with GPU support can be found below.

- dicom2niix: If you plan to process raw DICOM data.

You require a Docker installation for runnign BRATS models as well as Glioblstoma models.

Any remaining dependencies are Python packages, such as brainles-preprocessing, brats and antspy, are listed in requirements.txt. They are installed alongside the Python package explained in the following section.


## Installation

```bash
git clone https://github.com/LMZimmer/gbm_bench.git
cd gbm_bench
pip install .
```

## Features
- Standardized processing pipeline for Brain MRIs
- Extensible benchmark framework for dockered Glioblastoma models
- Easy-to-use minimal API


### Docker and NVIDIA Container Toolkit Setup

- **Docker**: Installation instructions on the official [website](https://docs.docker.com/get-docker/)
- **NVIDIA Container Toolkit**: Refer to the [NVIDIA install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and the official [GitHub page](https://github.com/NVIDIA/nvidia-container-toolkit)

