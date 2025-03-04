# First install libmamba solver in base environment
conda install -n base conda-libmamba-solver

# Then activate our environment
conda activate geograph

# Now install RAPIDS with the libmamba solver
conda install --solver=libmamba -c rapidsai -c conda-forge -c nvidia \
    rapids=25.02 'cuda-version>=12.0,<=12.8' \
    graphistry jupyterlab networkx nx-cugraph=25.02 dash xarray-spatial -y


conda install --solver=libmamba -c nvidia cuda-cudart cuda-version=12