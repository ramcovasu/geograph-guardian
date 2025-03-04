"""Weather module for GeoGraph Guardian.

This module provides weather data integration and analysis capabilities for supply chain risk management.

Components:
- weather_service.py: OpenWeatherMap API client
- geo_mapping.py: Geographic mapping utility for suppliers
- schema.py: Weather database schema definition
- weather_data_manager.py: Weather data collection and analysis
- weather_impact_ui.py: Streamlit UI for weather impact analysis
"""

# Import key components for easier access
from src.weather.weather_service import WeatherService
from src.weather.geo_mapping import GeoMapper
from src.weather.schema import WeatherDatabaseSchema
from src.weather.weather_data_manager import WeatherDataManager