# GBM_bench
Glioblastoma Model Benchmark (working title)


 1. Create a virtual env with python <3.11: conda create -n brainles python=3.10
# conda create -n brainles python=3.10
# source activate base
# conda activate brainles

# 2. Get dicom2niix and adapt the script path in 1_

# 3. Install python packages
pip install brainles-preprocessing
pip install brainles_preprocessing[ereg]
pip install dipy
pip install PyPDF2
pip install brats

# 4. Set up docker
On windows, Docker Desktop should be sufficient. On Linux, Docker and the NVIDIA Container Toolkit need to be installed.

# 5. Download docker algos
This should happen when using BRATS for the first time.

# 6. Install SynthSeg
Follow https://github.com/BBillot/SynthSeg (clone, set up versions, download models)

# 7. Install ANTS
Follow https://github.com/ANTsX/ANTs (download binaries, extract, add to PATH)




# Editable installation
pip install -e .
