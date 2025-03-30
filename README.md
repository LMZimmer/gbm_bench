# GBMbench
Glioblastoma Model Benchmark (working title)


## Prerequisites

dicom2niix, brainles-preprocessing (installs antspy), brats, dipy, PyPDF2, docker, python (<3.11 from brainles?)


## Installation

```bash
git clone https://github.com/LMZimmer/gbm_bench.git
cd gbm_bench
pip install .
```

## Features
- Standardized processing pipeline for Brain MRIs
- Easy-to-use minimal API


> [!IMPORTANT]  
> To run GBMBench you require a Docker installation. <br>
> Some algorithms also require GPU support (NVIDIA Docker). <br>


### Docker and NVIDIA Container Toolkit Setup

- **Docker**: Installation instructions on the official [website](https://docs.docker.com/get-docker/)
- **NVIDIA Container Toolkit**: Refer to the [NVIDIA install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and the official [GitHub page](https://github.com/NVIDIA/nvidia-container-toolkit)

