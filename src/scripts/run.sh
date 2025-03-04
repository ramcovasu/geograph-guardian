#!/bin/bash

# Get absolute paths
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
SCRIPT_DIR="$PROJECT_ROOT/src/scripts"
DATA_DIR="$PROJECT_ROOT/data"
PROCESSED_DIR="$DATA_DIR/processed"
RAW_DIR="$DATA_DIR/raw"
LOG_DIR="$PROJECT_ROOT/logs"

echo "=== GeoGraph Guardian Data Pipeline ==="
echo "Starting pipeline execution at $(date)"

# Setup directories
echo "Setting up directories..."
mkdir -p "$PROCESSED_DIR" "$RAW_DIR" "$LOG_DIR"

# Set up Python path (only once)
# Clear any existing PYTHONPATH first
unset PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT"
echo "PYTHONPATH set to: $PYTHONPATH"

# Print directory structure for debugging
echo "Project structure:"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "Current directory: $(pwd)"
ls -R "$PROJECT_ROOT/src"

# Activate conda environment
if command -v conda &> /dev/null; then
    echo "Activating conda environment 'geograph'..."
    # Source conda for this shell session
    CONDA_PATH=$(conda info --base)
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    conda activate geograph
else
    echo "ERROR: conda not found. Please install conda first."
    exit 1
fi

# Clean up processed folder
echo -e "\n=== Cleaning up processed folder ==="
if [ -d "$PROCESSED_DIR" ]; then
    echo "Removing existing processed files..."
    rm -rf "$PROCESSED_DIR"/*
    echo "Processed folder cleaned"
fi

# Check if ArangoDB is running
echo -e "\n=== Checking Database Service ==="
if systemctl is-active --quiet arangodb3; then
    echo "ArangoDB is running"
else
    echo "Starting ArangoDB service..."
    sudo systemctl start arangodb3
    sleep 5  # Give ArangoDB time to start
fi

# Verify Python imports with better error handling
echo -e "\n=== Verifying Python imports ==="
IMPORT_TEST=$(python3 -c "
try:
    from src.data_processing.arango_ingestor import get_ingestion_configs
    print('SUCCESS')
except ImportError as e:
    print(f'IMPORT_ERROR: {str(e)}')
except Exception as e:
    print(f'ERROR: {str(e)}')
" 2>&1)

if [[ $IMPORT_TEST == *"SUCCESS"* ]]; then
    echo "Python imports verified successfully"
else
    echo "ERROR: Python imports failed:"
    echo "$IMPORT_TEST"
    echo "Current PYTHONPATH: $PYTHONPATH"
    echo "Please check your module structure:"
    echo "1. Ensure you have __init__.py in all directories"
    echo "2. Check file permissions"
    echo "3. Verify module structure matches imports"
    exit 1
fi

# Ask for confirmation before proceeding
read -p "This will clear all existing data in ArangoDB. Do you want to continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled"
    exit 1
fi

# Step 1: Data Processing
echo -e "\n=== Running Data Processing Pipeline ==="
echo "Processing raw data files..."
PYTHONPATH="$PROJECT_ROOT" python3 "$SCRIPT_DIR/process_data.py"

# Check processing success
if [ $? -eq 0 ]; then
    echo "Data processing completed successfully"

    # Step 2: Data Ingestion
    echo -e "\n=== Running Data Ingestion Pipeline ==="
    echo "Ingesting processed data into ArangoDB..."
    PYTHONPATH="$PROJECT_ROOT" python3 "$SCRIPT_DIR/ingest_data.py"

    if [ $? -eq 0 ]; then
        echo "Data ingestion completed successfully"
        
        # Step 3: Data Validation
        echo -e "\n=== Running Data Validation ==="
        echo "Validating data in ArangoDB..."
        PYTHONPATH="$PROJECT_ROOT" python3 "$SCRIPT_DIR/validate_data.py"
        
        VALIDATION_RESULT=$?
        if [ $VALIDATION_RESULT -eq 0 ]; then
            echo "Data validation completed successfully"
            echo -e "\n=== Pipeline Execution Complete ==="
            echo "Finished at $(date)"
            exit 0
        else
            echo "WARNING: Data validation had some issues"
            echo "Check the validation logs for details"
            exit 1
        fi
    else
        echo "ERROR: Data ingestion failed"
        exit 1
    fi
else
    echo "ERROR: Data processing failed"
    exit 1
fi