# GeoGraph Guardian

GeoGraph Guardian is an advanced supply chain risk monitoring system that combines graph analytics with real-time geopolitical data and AI to transform supply chain risk management.

## Overview

GeoGraph Guardian enables:

- **Real-time risk prediction** by analyzing network patterns and geopolitical events
- **Natural language queries** for complex supply chain scenarios
- **GPU-accelerated graph analytics** to identify vulnerable nodes and alternative paths
- **Hybrid query execution** combining ArangoDB's AQL for path analysis and cuGraph for complex network metrics
- **Interactive visualizations** showing risk propagation through the supply chain network
- **Automated mitigation recommendations** based on historical patterns and network structure

## Features

- **Graph-powered Supply Chain Analytics**: Visualize and analyze your entire supply chain as a graph network.
- **Natural Language Interface**: Ask complex questions about your supply chain in plain English.
- **Weather Impact Analysis**: Monitor and assess weather-related risks to your supply chain.
- **AI-Driven Insights**: Receive AI-generated explanations and recommendations for risk mitigation.
- **GPU Acceleration**: Process large supply chain networks with NVIDIA GPU acceleration.

## Architecture

GeoGraph Guardian is built with a sophisticated tech stack:

- **Data Integration**: Combines supply chain networks with geopolitical event data
- **Graph Processing**: Uses NetworkX for data transformation and ArangoDB for persistent storage
- **GPU Acceleration**: Leverages NVIDIA's cuGraph for complex network algorithms like community detection and centrality analysis
- **AI Agent Framework**: Built with LangGraph and LangChain for natural language processing and response generation
- **Hybrid Query System**: Dynamically routes queries between AQL and cuGraph

## Installation

### Prerequisites

- Python 3.10+ 
- NVIDIA GPU with CUDA 12.0+ (for GPU acceleration)
- ArangoDB 3.11+

### Configuration

1. Configure ArangoDB connection by updating the `config/arangodb.yaml` file:
   ```yaml
   development:
     host: 127.0.0.1
     port: 8529
     database: geograph
     username: root
     password: your_password
     graph_name: supplychain
   ```

2. Ensure the collections configuration in the same file matches your data model:
   ```yaml
   collections:
     vertices:
       - name: suppliers
       - name: parts
       - name: products
       # Other vertex collections...
     edges:
       - name: supplier_provides_part
       - name: part_depends_on
       # Other edge collections...
   ```

### Basic Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ramcovasu/Geograph.git
   cd geograph-guardian
   ```

2. Run the main installation script:
   ```bash
   bash install.sh
   ```
   This script will:
   - Check and install CUDA if needed
   - Set up a Conda environment
   - Install RAPIDS and other dependencies
   - Install ArangoDB

3. Alternatively, you can install dependencies manually:
   ```bash
   pip install -r requirements.txt
   ```

4. Run additional setup if needed:
   ```bash
   bash setup_additional.sh
   ```
   This handles starting ArangoDB and installing additional packages.

> **Note:** The installation scripts (`install.sh`, `rapids.sh`, `setup_additional.sh`) can be moved into the project directory structure for better organization. Create an `install` directory within the project and move these files there.

### GPU Acceleration (Optional)

For GPU acceleration with NVIDIA GPUs:

```bash
bash rapids.sh
```

### Weather Module (Optional)

To enable the weather impact analysis feature:

1. Get an API key from [OpenWeatherMap](https://openweathermap.org/api)
2. Install the weather module:
   ```bash
   bash install_weather_module.sh
   ```
3. Add your API key to the `.env` file:
   ```
   OPENWEATHER_API_KEY=your_api_key_here
   ```

## Project Structure

After organizing the installation files, your project structure should look like this:

```
geograph-guardian/
├── app.py                  # Main Streamlit application
├── config/
│   └── arangodb.yaml       # Database configuration
├── data/
│   ├── cache/              # Cache for weather data
│   ├── processed/          # Processed data files
│   ├── raw/                # Raw data files
│   └── reference/          # Reference data
├── install/                # Installation scripts
│   ├── install.sh          # Main installation script
│   ├── rapids.sh           # RAPIDS installation
│   ├── Release.key         # ArangoDB release key
│   └── setup_additional.sh # Additional setup steps
├── logs/                   # Application logs
├── requirements.txt        # Python dependencies
├── run_streamlit.sh        # Script to run the Streamlit app
└── src/                    # Source code
    ├── data_processing/    # Data processing modules
    ├── graph_analytics/    # Graph analytics algorithms
    ├── llm/                # LLM integration
    ├── scripts/            # Utility scripts
    ├── streamlit/          # Streamlit UI components
    ├── utils/              # Utility functions
    ├── visualization/      # Visualization modules
    └── weather/            # Weather analysis module
```

## Data Setup

To initialize the database and load sample data:

```bash
# Process and ingest data
python src/scripts/process_data.py
python src/scripts/ingest_data.py

# Verify data ingestion
python src/scripts/validate_data.py
```

Alternatively, use the all-in-one script:

```bash
bash run.sh
```

## Usage

Start the application using the provided script:

```bash
bash run_streamlit.sh
```

Or manually with:

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501 with three main interfaces:

1. **Chat Assistant**: Ask natural language questions about your supply chain
2. **Graph Analytics**: Perform advanced graph analytics such as community detection, centrality analysis, and shortest path analysis
3. **Weather Impact Analysis**: Monitor real-time weather impacts on your supply chain

## Example Queries

Here are some example questions you can ask the chat assistant:

1. "Can you get 7 suppliers and their risk scores?"
2. "Can you give me the name of the suppliers and their risk scores who can supply parts similar to the suppliers in JAPAN?"
3. "Show me suppliers who have had delayed purchase orders and their current risk scores"
4. "Show parts with HIGH criticality level where current inventory is below safety stock, include the supplier's risk score and lead time"
5. "List all warehouse locations where product part LID001 has quantity on hand greater than safety stock?"
6. "Show me all parts that had negative inventory transactions (issues/stockouts) along with their primary suppliers' risk scores and current inventory levels"
7. "Show me all suppliers who provide parts with HIGH criticality that are primary suppliers and have risk scores above .75"
8. "Can you give me names of alternate supplier to POWERCELL along with their risk scores and names of parts they can supply?"

## System Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB
- OS: Ubuntu 22.04, Windows 10+, or macOS 12+

### Recommended for GPU Acceleration
- NVIDIA GPU with 8GB+ VRAM
- CUDA Toolkit 12.0+
- 16GB+ System RAM

## License

[MIT License](LICENSE)

## Acknowledgments

- OpenWeatherMap API for real-time weather data
- NVIDIA RAPIDS team for GPU-accelerated data science libraries
- ArangoDB team for the graph database system