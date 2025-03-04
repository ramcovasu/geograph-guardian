#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Add project root to PYTHONPATH
export PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH

# Activate conda environment if needed
# conda activate geograph

# Run the Streamlit app
streamlit run src/streamlit/app.py