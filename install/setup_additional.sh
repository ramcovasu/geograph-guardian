sudo systemctl start arangodb3
sudo systemctl enable arangodb3  # to start on boot


conda activate geograph
pip install langchain langgraph


# Install Jupyter for development
conda install jupyter jupyterlab