#!/bin/bash

echo "=== Checking CUDA Installation ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA driver not found. Please install NVIDIA driver first."
    exit 1
fi

echo "=== NVIDIA Driver Information ==="
nvidia-smi

if ! command -v nvcc &> /dev/null; then
    echo "CUDA compiler not found. Installing CUDA Toolkit..."
    # Add NVIDIA package repositories
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
    sudo apt-get update
    sudo apt-get -y install cuda-12-6
fi

echo "=== CUDA Compiler Version ==="
nvcc --version

# Create a new conda environment
echo "=== Creating Conda Environment ==="
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

conda create -n geograph python=3.10 -y
conda activate geograph

# Install RAPIDS
echo "=== Installing RAPIDS ==="
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=23.12 cugraph=23.12 \
    python=3.10 cudatoolkit=12.6 -y

# Install other required packages
echo "=== Installing Core Python Packages ==="
conda install -c conda-forge \
    pandas \
    numpy \
    networkx \
    jupyter \
    jupyterlab \
    ipykernel \
    plotly \
    streamlit -y

echo "=== Installing AI/LLM Framework ==="
pip install "langchain>=0.1.0" "langchain-experimental>=0.0.47"
pip install "langgraph>=0.0.15"
pip install python-arango

# Install ArangoDB
echo "=== Installing ArangoDB ==="
curl -OL https://download.arangodb.com/arangodb311/DEBIAN/Release.key
sudo apt-key add - < Release.key
echo 'deb https://download.arangodb.com/arangodb311/DEBIAN/ /' | sudo tee /etc/apt/sources.list.d/arangodb.list
sudo apt-get update
sudo apt-get install arangodb3 -y

# Verify installations
echo "=== Verifying Installations ==="
python -c "import cugraph; import cudf; print('RAPIDS verification successful!')" || echo "WARNING: RAPIDS verification failed"
python -c "import networkx; import pandas; import numpy; import langchain; print('Core packages verification successful!')" || echo "WARNING: Core packages verification failed"

# Create Jupyter kernel
echo "=== Creating Jupyter Kernel ==="
python -m ipykernel install --user --name geograph --display-name "Python (GeoGraph)"

echo "=== Installation Complete! ==="
echo "Next steps:"
echo "1. Start ArangoDB service: sudo systemctl start arangodb3"
echo "2. Activate environment: conda activate geograph"
echo "3. Test installation by running Jupyter: jupyter lab"

# Add CUDA paths to .bashrc if not already present
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
    echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >> ~/.bashrc
fi

echo "Please source your .bashrc or restart your terminal:"
echo "source ~/.bashrc"