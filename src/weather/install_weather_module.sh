#!/bin/bash

echo "=== GeoGraph Guardian Weather Module Installation ==="
echo "Starting installation at $(date)"

# Get the project root
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
echo "Project root: $PROJECT_ROOT"

# Setup Python path
export PYTHONPATH="$PROJECT_ROOT"
echo "PYTHONPATH set to: $PYTHONPATH"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    echo "Activating conda environment 'geograph'..."
    # Source conda for this shell session
    CONDA_PATH=$(conda info --base)
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    conda activate geograph
else
    echo "Conda not found. Assuming you're using a different virtual environment."
fi

# Install required packages
echo "=== Installing Required Packages ==="
pip install requests plotly

# Create necessary directories and files
echo "=== Initializing Weather Module Structure ==="
python "$PROJECT_ROOT/src/weather/init_weather.py"

# Create .env file with placeholder for API key if it doesn't exist
ENV_FILE="$PROJECT_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "=== Creating .env file ==="
    echo "# OpenWeatherMap API Key" > "$ENV_FILE"
    echo "OPENWEATHER_API_KEY=your_api_key_here" >> "$ENV_FILE"
    echo "Created .env file with OpenWeatherMap API key placeholder"
else
    # Check if OPENWEATHER_API_KEY is already in .env
    if ! grep -q "OPENWEATHER_API_KEY" "$ENV_FILE"; then
        echo "" >> "$ENV_FILE"
        echo "# OpenWeatherMap API Key" >> "$ENV_FILE"
        echo "OPENWEATHER_API_KEY=your_api_key_here" >> "$ENV_FILE"
        echo "Added OpenWeatherMap API key placeholder to .env file"
    else
        echo "OpenWeatherMap API key already in .env file"
    fi
fi

# Setup the database schema
echo "=== Setting Up Weather Database Schema ==="
python "$PROJECT_ROOT/src/weather/setup_weather_db.py"

echo "=== Installation Complete ==="
echo "To use the Weather Impact Analysis module:"
echo "1. Get an API key from https://openweathermap.org/api"
echo "2. Add your API key to the .env file: OPENWEATHER_API_KEY=your_api_key"
echo "3. Restart the application"
echo ""
echo "Installation completed at $(date)"