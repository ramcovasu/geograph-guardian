import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import time
from pathlib import Path
import pandas as pd
import traceback
from dotenv import load_dotenv

from src.utils.logging import Logger


class WeatherService:
    """Client for retrieving weather data from OpenWeatherMap API."""
    
    def __init__(self):
        """Initialize the weather service with API credentials and cache settings."""
        load_dotenv()
        self.logger = Logger().get_logger()
        
        # Load API key from environment
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            self.logger.warning("OPENWEATHER_API_KEY not found. Weather service will not function.")
        
        # API endpoints
        self.current_weather_url = "https://api.openweathermap.org/data/2.5/weather"
        self.forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
        self.geolocation_url = "https://api.openweathermap.org/geo/1.0/direct"
        
        # Cache settings
        self.cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache' / 'weather'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_duration = timedelta(hours=1)  # Cache data for 1 hour
        
        # Weather event type mapping
        self.weather_event_mapping = {
            # Thunderstorm
            '2': 'thunderstorm',
            # Drizzle
            '3': 'drizzle',
            # Rain
            '5': 'rain',
            # Snow
            '6': 'snow',
            # Atmosphere (fog, haze, etc.)
            '7': 'atmosphere',
            # Clear
            '800': 'clear',
            # Clouds
            '8': 'clouds'
        }
        
        # Severity mapping (based on weather code intensity)
        self.severity_mapping = {
            # General pattern: higher second digit = more severe
            # Thunderstorm
            '200': 1, '201': 2, '202': 3, '210': 2, '211': 3, '212': 4, '221': 3, '230': 2, '231': 3, '232': 4,
            # Drizzle
            '300': 1, '301': 1, '302': 2, '310': 2, '311': 2, '312': 3, '313': 3, '314': 3, '321': 2,
            # Rain
            '500': 1, '501': 2, '502': 3, '503': 4, '504': 5, '511': 3, '520': 2, '521': 3, '522': 4, '531': 3,
            # Snow
            '600': 2, '601': 3, '602': 4, '611': 3, '612': 3, '613': 4, '615': 2, '616': 3, '620': 2, '621': 3, '622': 4,
            # Atmosphere
            '701': 1, '711': 2, '721': 1, '731': 2, '741': 1, '751': 2, '761': 2, '762': 3, '771': 4, '781': 5,
            # Clear
            '800': 0,
            # Clouds
            '801': 0, '802': 0, '803': 1, '804': 1
        }
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the path to a cache file."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Try to get data from cache if it exists and is not expired."""
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is expired
            cached_time = datetime.fromisoformat(cached_data['cached_at'])
            if datetime.now() - cached_time > self.cache_duration:
                self.logger.debug(f"Cache expired for {cache_key}")
                return None
            
            self.logger.debug(f"Retrieved from cache: {cache_key}")
            return cached_data['data']
        except Exception as e:
            self.logger.warning(f"Error reading cache file {cache_key}: {str(e)}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        """Save data to cache with timestamp."""
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_data = {
                'cached_at': datetime.now().isoformat(),
                'data': data
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            
            self.logger.debug(f"Saved to cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Error writing to cache file {cache_key}: {str(e)}")
    
    def get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """Get coordinates (latitude, longitude) for a location name."""
        if not self.api_key:
            self.logger.error("Cannot get coordinates: API key not configured")
            return None
        
        cache_key = f"geo_{location.replace(' ', '_').lower()}"
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data:
            if cached_data and len(cached_data) > 0:
                return cached_data[0]['lat'], cached_data[0]['lon']
            return None
        
        params = {
            'q': location,
            'limit': 1,
            'appid': self.api_key
        }
        
        try:
            response = requests.get(self.geolocation_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            self._save_to_cache(cache_key, data)
            
            if data and len(data) > 0:
                return data[0]['lat'], data[0]['lon']
            
            self.logger.warning(f"No coordinates found for location: {location}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting coordinates for {location}: {str(e)}")
            return None
    
    def get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """Get current weather for specified coordinates."""
        if not self.api_key:
            self.logger.error("Cannot get current weather: API key not configured")
            return None
        
        cache_key = f"current_{lat}_{lon}"
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data:
            return cached_data
        
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'  # Use metric units
        }
        
        try:
            response = requests.get(self.current_weather_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            self._save_to_cache(cache_key, data)
            
            return data
        except Exception as e:
            self.logger.error(f"Error getting current weather for ({lat}, {lon}): {str(e)}")
            return None
    
    def get_forecast(self, lat: float, lon: float) -> Optional[Dict]:
        """Get 5-day forecast for specified coordinates."""
        if not self.api_key:
            self.logger.error("Cannot get forecast: API key not configured")
            return None
        
        cache_key = f"forecast_{lat}_{lon}"
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data:
            return cached_data
        
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'  # Use metric units
        }
        
        try:
            response = requests.get(self.forecast_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            self._save_to_cache(cache_key, data)
            
            return data
        except Exception as e:
            self.logger.error(f"Error getting forecast for ({lat}, {lon}): {str(e)}")
            return None
    
    def get_weather_by_location(self, location: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Get current weather and forecast for a location name."""
        coordinates = self.get_coordinates(location)
        
        if not coordinates:
            return None, None
        
        lat, lon = coordinates
        current = self.get_current_weather(lat, lon)
        forecast = self.get_forecast(lat, lon)
        
        return current, forecast
    
    def extract_weather_events(self, weather_data: Dict, forecast_data: Optional[Dict] = None) -> List[Dict]:
        """
        Extract structured weather events from weather data and optional forecast data.
        
        Args:
            weather_data: Current weather data
            forecast_data: Optional forecast data
            
        Returns:
            List of weather events with standardized structure
        """
        events = []
        
        # FIX: Add validation and debug logging to diagnose issues
        try:
            # Process current weather
            if weather_data and isinstance(weather_data, dict) and 'weather' in weather_data:
                weather_list = weather_data.get('weather', [])
                
                # Debug log the data types
                self.logger.info(f"Weather list type: {type(weather_list)}")
                if weather_list and len(weather_list) > 0:
                    self.logger.info(f"First weather item type: {type(weather_list[0])}")
                
                # Handle cases where weather list might be a string
                if isinstance(weather_list, str):
                    self.logger.warning(f"Weather list is a string instead of a list: {weather_list}")
                    try:
                        # Try to parse it as JSON
                        weather_list = json.loads(weather_list)
                    except:
                        self.logger.error(f"Failed to parse weather list as JSON: {weather_list}")
                        weather_list = []
                
                for weather_item in weather_list:
                    # Handle case where weather_item might be a string
                    if isinstance(weather_item, str):
                        self.logger.warning(f"Weather item is a string instead of a dict: {weather_item}")
                        try:
                            # Try to parse it as JSON
                            weather_item = json.loads(weather_item)
                        except:
                            self.logger.error(f"Failed to parse weather item as JSON: {weather_item}")
                            continue
                    
                    if not isinstance(weather_item, dict):
                        self.logger.warning(f"Weather item is not a dict: {weather_item}")
                        continue
                        
                    event = self._create_weather_event(
                        weather_item,
                        location_name=weather_data.get('name', 'Unknown'),
                        country_code=weather_data.get('sys', {}).get('country', ''),
                        timestamp=weather_data.get('dt', 0),
                        coordinates=(weather_data.get('coord', {}).get('lat', 0), 
                                    weather_data.get('coord', {}).get('lon', 0))
                    )
                    events.append(event)
            else:
                self.logger.warning(f"Invalid weather data format: {type(weather_data)}")
        
            # Process forecast data if available
            if forecast_data and isinstance(forecast_data, dict) and 'list' in forecast_data:
                city_name = forecast_data.get('city', {}).get('name', 'Unknown')
                country_code = forecast_data.get('city', {}).get('country', '')
                coord_lat = forecast_data.get('city', {}).get('coord', {}).get('lat', 0)
                coord_lon = forecast_data.get('city', {}).get('coord', {}).get('lon', 0)
                
                forecast_list = forecast_data.get('list', [])
                
                # Handle cases where forecast list might be a string
                if isinstance(forecast_list, str):
                    self.logger.warning(f"Forecast list is a string instead of a list: {forecast_list}")
                    try:
                        # Try to parse it as JSON
                        forecast_list = json.loads(forecast_list)
                    except:
                        self.logger.error(f"Failed to parse forecast list as JSON: {forecast_list}")
                        forecast_list = []
                
                for forecast_item in forecast_list:
                    # Handle case where forecast_item might be a string
                    if isinstance(forecast_item, str):
                        self.logger.warning(f"Forecast item is a string instead of a dict: {forecast_item}")
                        try:
                            # Try to parse it as JSON
                            forecast_item = json.loads(forecast_item)
                        except:
                            self.logger.error(f"Failed to parse forecast item as JSON: {forecast_item}")
                            continue
                    
                    if not isinstance(forecast_item, dict):
                        self.logger.warning(f"Forecast item is not a dict: {forecast_item}")
                        continue
                        
                    weather_list = forecast_item.get('weather', [])
                    
                    # Handle cases where weather list might be a string
                    if isinstance(weather_list, str):
                        self.logger.warning(f"Forecast weather list is a string: {weather_list}")
                        try:
                            # Try to parse it as JSON
                            weather_list = json.loads(weather_list)
                        except:
                            self.logger.error(f"Failed to parse forecast weather list: {weather_list}")
                            weather_list = []
                    
                    for weather_item in weather_list:
                        # Handle case where weather_item might be a string
                        if isinstance(weather_item, str):
                            self.logger.warning(f"Forecast weather item is a string: {weather_item}")
                            try:
                                # Try to parse it as JSON
                                weather_item = json.loads(weather_item)
                            except:
                                self.logger.error(f"Failed to parse forecast weather item: {weather_item}")
                                continue
                        
                        if not isinstance(weather_item, dict):
                            self.logger.warning(f"Forecast weather item is not a dict: {weather_item}")
                            continue
                            
                        event = self._create_weather_event(
                            weather_item,
                            location_name=city_name,
                            country_code=country_code,
                            timestamp=forecast_item.get('dt', 0),
                            coordinates=(coord_lat, coord_lon),
                            is_forecast=True,
                            forecast_item=forecast_item
                        )
                        events.append(event)
            elif forecast_data:
                self.logger.warning(f"Invalid forecast data format: {type(forecast_data)}")
                
        except Exception as e:
            self.logger.error(f"Error extracting weather events: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        return events
    
    def _create_weather_event(self, 
                            weather_item: Dict, 
                            location_name: str, 
                            country_code: str,
                            timestamp: int,
                            coordinates: Tuple[float, float],
                            is_forecast: bool = False,
                            forecast_item: Optional[Dict] = None) -> Dict:
        """
        Create a standardized weather event object from API data.
        
        Args:
            weather_item: Weather condition item from API
            location_name: Name of the location
            country_code: Two-letter country code
            timestamp: Unix timestamp
            coordinates: (latitude, longitude) tuple
            is_forecast: Whether this is a forecast or current weather
            forecast_item: Full forecast item if is_forecast is True
            
        Returns:
            Standardized weather event dictionary
        """
        try:
            # FIX: Add validation for weather_item
            if not isinstance(weather_item, dict):
                self.logger.error(f"Weather item is not a dictionary: {weather_item}")
                # Convert to dict if it's a string
                if isinstance(weather_item, str):
                    try:
                        weather_item = json.loads(weather_item)
                    except:
                        # Create a minimal placeholder dictionary
                        weather_item = {'id': 800, 'description': 'unknown'}
            
            # Get weather code and extract event type and severity
            weather_id = str(weather_item.get('id', 0))
            
            # Determine event type based on first digit or the full code
            event_type = None
            for prefix, event in self.weather_event_mapping.items():
                if weather_id.startswith(prefix) or weather_id == prefix:
                    event_type = event
                    break
            
            if not event_type:
                event_type = 'unknown'
            
            # Determine severity (1-5 scale)
            severity = self.severity_mapping.get(weather_id, 0)
            
            # Create event object
            event = {
                'event_id': f"weather_{timestamp}_{weather_id}",
                'event_type': event_type,
                'weather_id': weather_id,
                'description': weather_item.get('description', ''),
                'severity': severity,
                'location_name': location_name,
                'country_code': country_code,
                'coordinates': {
                    'lat': coordinates[0],
                    'lon': coordinates[1]
                },
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).isoformat(),
                'is_forecast': is_forecast
            }
            
            # Add forecast-specific data if available
            if is_forecast and forecast_item and isinstance(forecast_item, dict):
                # FIX: Validate forecast_item and access nested data safely
                main_data = forecast_item.get('main', {})
                if isinstance(main_data, str):
                    try:
                        main_data = json.loads(main_data)
                    except:
                        main_data = {}
                
                wind_data = forecast_item.get('wind', {})
                if isinstance(wind_data, str):
                    try:
                        wind_data = json.loads(wind_data)
                    except:
                        wind_data = {}
                
                event.update({
                    'temperature': main_data.get('temp') if isinstance(main_data, dict) else None,
                    'feels_like': main_data.get('feels_like') if isinstance(main_data, dict) else None,
                    'pressure': main_data.get('pressure') if isinstance(main_data, dict) else None,
                    'humidity': main_data.get('humidity') if isinstance(main_data, dict) else None,
                    'wind_speed': wind_data.get('speed') if isinstance(wind_data, dict) else None,
                    'wind_direction': wind_data.get('deg') if isinstance(wind_data, dict) else None,
                    'precipitation_probability': forecast_item.get('pop', 0)
                })
            elif not is_forecast and 'main' in weather_item:
                # FIX: For current weather, validate main data
                main_data = weather_item.get('main')
                if isinstance(main_data, str):
                    try:
                        main_data = json.loads(main_data)
                    except:
                        main_data = {}
                        
                # Add current weather metrics
                event.update({
                    'temperature': main_data.get('temp') if isinstance(main_data, dict) else None,
                    'feels_like': main_data.get('feels_like') if isinstance(main_data, dict) else None,
                    'pressure': main_data.get('pressure') if isinstance(main_data, dict) else None,
                    'humidity': main_data.get('humidity') if isinstance(main_data, dict) else None,
                    'wind_speed': weather_item.get('wind', {}).get('speed') if isinstance(weather_item.get('wind', {}), dict) else None,
                    'wind_direction': weather_item.get('wind', {}).get('deg') if isinstance(weather_item.get('wind', {}), dict) else None
                })
            
            return event
            
        except Exception as e:
            self.logger.error(f"Error creating weather event: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Return a minimal valid event
            return {
                'event_id': f"weather_error_{timestamp}",
                'event_type': 'unknown',
                'weather_id': '800',
                'description': 'Error creating weather event',
                'severity': 0,
                'location_name': location_name,
                'country_code': country_code,
                'coordinates': {
                    'lat': coordinates[0],
                    'lon': coordinates[1]
                },
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).isoformat(),
                'is_forecast': is_forecast
            }
    
    def get_severe_weather_events(self, location: str, min_severity: int = 3) -> List[Dict]:
        """
        Get severe weather events for a location.
        
        Args:
            location: Location name (city, country, etc.)
            min_severity: Minimum severity level (1-5)
            
        Returns:
            List of severe weather events
        """
        current, forecast = self.get_weather_by_location(location)
        all_events = self.extract_weather_events(current, forecast)
        
        # Filter for severe events
        severe_events = [event for event in all_events if event['severity'] >= min_severity]
        
        return severe_events
    
    def get_weather_events_by_coords(self, lat: float, lon: float, min_severity: int = 0) -> List[Dict]:
        """
        Get weather events for specified coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            min_severity: Minimum severity level (1-5)
            
        Returns:
            List of weather events
        """
        current = self.get_current_weather(lat, lon)
        forecast = self.get_forecast(lat, lon)
        
        # Debug log the API responses
        self.logger.info(f"Current weather response type: {type(current)}")
        self.logger.info(f"Forecast response type: {type(forecast)}")
        
        all_events = self.extract_weather_events(current, forecast)
        
        if min_severity > 0:
            return [event for event in all_events if event['severity'] >= min_severity]
        
        return all_events