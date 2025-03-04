import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime, timedelta
import json
import re
import traceback

from src.utils.logging import Logger
from src.utils.db_connection import ArangoDB
from src.weather.weather_service import WeatherService
from src.weather.geo_mapping import GeoMapper


class WeatherDataManager:
    """Manages weather data collection, storage, and analysis."""
    
    def __init__(self):
        """Initialize the WeatherDataManager with required dependencies."""
        self.logger = Logger().get_logger()
        self.db = ArangoDB()
        self.weather_service = WeatherService()
        self.geo_mapper = GeoMapper()
    
    def update_supplier_locations(self) -> int:
        """
        Update supplier locations in the database.
        
        Returns:
            Number of suppliers updated
        """
        return self.geo_mapper.update_supplier_locations()
    
    def collect_weather_for_suppliers(self, severity_threshold: int = 0) -> Tuple[int, int]:
        """
        Collect weather data for all supplier locations.
        
        Args:
            severity_threshold: Minimum severity to save weather events (0-5)
            
        Returns:
            Tuple of (suppliers_processed, events_stored)
        """
        try:
            # Get all supplier locations
            query = """
            FOR loc IN supplier_locations
            RETURN loc
            """
            
            cursor = self.db.db.aql.execute(query)
            locations = list(cursor)
            
            self.logger.info(f"Processing weather for {len(locations)} supplier locations")
            
            events_stored = 0
            suppliers_processed = 0
            
            for location in locations:
                try:
                    # Check if location is a dict or string
                    if isinstance(location, str):
                        self.logger.warning(f"Location is a string, not a dict: {location}")
                        continue

                    # Extract coordinates and do additional validation
                    supplier_id = location.get('supplier_id')
                    coordinates = location.get('coordinates')
                    
                    self.logger.info(f"Processing supplier {supplier_id} with coordinates type: {type(coordinates)}")
                    
                    # Handle different types of coordinates
                    if coordinates is None:
                        self.logger.warning(f"Missing coordinates for supplier {supplier_id}")
                        continue
                        
                    # Handle case where coordinates are stored as a string
                    if isinstance(coordinates, str):
                        self.logger.warning(f"Coordinates for supplier {supplier_id} stored as string: {coordinates}")
                        try:
                            # Try to parse as JSON
                            if '{' in coordinates:
                                coordinates = json.loads(coordinates.replace("'", '"'))
                            else:
                                # Try to extract using regex
                                lat_match = re.search(r'lat[\'"]?\s*:\s*([0-9.-]+)', coordinates)
                                lon_match = re.search(r'lon[\'"]?\s*:\s*([0-9.-]+)', coordinates)
                                
                                if lat_match and lon_match:
                                    coordinates = {
                                        'lat': float(lat_match.group(1)),
                                        'lon': float(lon_match.group(1))
                                    }
                                else:
                                    self.logger.error(f"Could not parse coordinates string: {coordinates}")
                                    continue
                        except Exception as e:
                            self.logger.error(f"Error parsing coordinates string for {supplier_id}: {str(e)}")
                            continue
                    
                    # Final validation of coordinates format
                    if not isinstance(coordinates, dict):
                        self.logger.error(f"Coordinates for {supplier_id} is not a dict after parsing: {coordinates}")
                        continue
                        
                    lat = coordinates.get('lat')
                    lon = coordinates.get('lon')
                    
                    if not lat or not lon:
                        self.logger.warning(f"Missing lat/lon in coordinates for supplier {supplier_id}: {coordinates}")
                        continue
                    
                    # Convert to float if they're strings
                    try:
                        if isinstance(lat, str):
                            lat = float(lat)
                        if isinstance(lon, str):
                            lon = float(lon)
                    except ValueError:
                        self.logger.error(f"Could not convert coordinates to float for {supplier_id}: {coordinates}")
                        continue
                    
                    self.logger.info(f"Getting weather for supplier {supplier_id} at coordinates: lat={lat}, lon={lon}")
                    
                    # Get weather events
                    events = self.weather_service.get_weather_events_by_coords(
                        lat=lat, 
                        lon=lon, 
                        min_severity=severity_threshold
                    )
                    
                    if not events:
                        self.logger.info(f"No significant weather events for supplier {supplier_id}")
                        suppliers_processed += 1
                        continue
                    
                    # Store weather events
                    stored_count = self._store_weather_events(events)
                    events_stored += stored_count
                    
                    # Connect suppliers to weather events
                    self._connect_supplier_to_events(supplier_id, events)
                    
                    suppliers_processed += 1
                    
                except Exception as e:
                    supplier_id = location.get('supplier_id') if isinstance(location, dict) else str(location)
                    self.logger.error(f"Error processing weather for supplier {supplier_id}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    # Continue processing other suppliers
                    continue
            
            self.logger.info(f"Processed {suppliers_processed} suppliers, stored {events_stored} weather events")
            return suppliers_processed, events_stored
            
        except Exception as e:
            self.logger.error(f"Error collecting weather for suppliers: {str(e)}")
            self.logger.error(traceback.format_exc())
            return 0, 0
    
    def _store_weather_events(self, events: List[Dict]) -> int:
        """
        Store weather events in the database.
        
        Args:
            events: List of weather event dictionaries
            
        Returns:
            Number of events stored
        """
        if not events:
            return 0
        
        stored_count = 0
        collection = self.db.db.collection('weather_events')
        
        for event in events:
            try:
                # Create document key from event ID
                event_doc = event.copy()
                event_doc['_key'] = event_doc['event_id']
                
                # Format coordinates as GeoJSON for spatial indexing
                if 'coordinates' in event_doc:
                    lat = event_doc['coordinates'].get('lat')
                    lon = event_doc['coordinates'].get('lon')
                    
                    # Keep the original coordinates but add GeoJSON format
                    event_doc['geo'] = {
                        'type': 'Point',
                        'coordinates': [lon, lat]  # GeoJSON format is [longitude, latitude]
                    }
                
                # Store or update the document
                collection.insert(event_doc, overwrite=True)
                stored_count += 1
                
            except Exception as e:
                self.logger.error(f"Error storing weather event {event.get('event_id')}: {str(e)}")
        
        return stored_count
    
    def _connect_supplier_to_events(self, supplier_id: str, events: List[Dict]) -> None:
        """
        Create edges connecting suppliers to weather events.
        
        Args:
            supplier_id: Supplier ID
            events: List of weather events
        """
        if not events:
            return
        
        edge_collection = self.db.db.collection('supplier_affected_by_weather')
        
        for event in events:
            try:
                event_id = event.get('event_id')
                
                # Create edge document
                edge_doc = {
                    '_from': f'suppliers/{supplier_id}',
                    '_to': f'weather_events/{event_id}',
                    'impact_score': self._calculate_impact_score(event),
                    'event_type': event.get('event_type'),
                    'severity': event.get('severity', 0),
                    'created_at': datetime.now().isoformat()
                }
                
                # Create a unique key for the edge
                edge_doc['_key'] = f"{supplier_id}_{event_id}"
                
                # Store the edge
                edge_collection.insert(edge_doc, overwrite=True)
                
            except Exception as e:
                self.logger.error(f"Error connecting supplier {supplier_id} to event {event.get('event_id')}: {str(e)}")
    
    def _calculate_impact_score(self, event: Dict) -> float:
        """
        Calculate an impact score for a weather event.
        
        Args:
            event: Weather event dictionary
            
        Returns:
            Impact score (0-100)
        """
        # Basic impact score calculation based on severity and forecast probability
        base_score = event.get('severity', 0) * 20  # Convert 0-5 scale to 0-100
        
        # Adjust by precipitation probability if available
        if 'precipitation_probability' in event:
            probability = event.get('precipitation_probability', 0)
            base_score = base_score * (0.5 + probability * 0.5)  # Scale by probability
        
        # Future enhancement: adjust by supplier criticality
        
        return min(100, max(0, base_score))  # Ensure score is between 0-100
    
    def get_severe_weather_events(self, min_severity: int = 3) -> List[Dict]:
        """
        Get all severe weather events from the database.
        
        Args:
            min_severity: Minimum severity (1-5)
            
        Returns:
            List of severe weather events
        """
        query = """
        FOR event IN weather_events
        FILTER event.severity >= @severity
        SORT event.severity DESC, event.timestamp ASC
        RETURN event
        """
        
        try:
            cursor = self.db.db.aql.execute(query, bind_vars={'severity': min_severity})
            return list(cursor)
        except Exception as e:
            self.logger.error(f"Error retrieving severe weather events: {str(e)}")
            return []
    
    def get_weather_for_country(self, country: str, min_severity: int = 0) -> List[Dict]:
        """
        Get weather events for a specific country.
        
        Args:
            country: Country name or code
            min_severity: Minimum severity (0-5)
            
        Returns:
            List of weather events
        """
        query = """
        FOR event IN weather_events
        FILTER UPPER(event.country_code) == UPPER(@country) OR 
               CONTAINS(UPPER(event.location_name), UPPER(@country))
        FILTER event.severity >= @severity
        SORT event.severity DESC, event.timestamp ASC
        RETURN event
        """
        
        try:
            cursor = self.db.db.aql.execute(query, bind_vars={
                'country': country,
                'severity': min_severity
            })
            return list(cursor)
        except Exception as e:
            self.logger.error(f"Error retrieving weather for country {country}: {str(e)}")
            return []
    
    def get_suppliers_affected_by_weather(self, min_severity: int = 3) -> List[Dict]:
        """
        Get suppliers affected by severe weather events.
        
        Args:
            min_severity: Minimum severity (1-5)
            
        Returns:
            List of suppliers with weather impact information
        """
        query = """
        FOR edge IN supplier_affected_by_weather
            LET supplier = DOCUMENT(edge._from)
            LET event = DOCUMENT(edge._to)
            FILTER event.severity >= @severity
            RETURN {
                supplier_id: supplier.supplier_id,
                supplier_name: supplier.supplier_name,
                country: supplier.country,
                region: supplier.region,
                event_id: event.event_id,
                event_type: event.event_type,
                severity: event.severity,
                description: event.description,
                timestamp: event.timestamp,
                datetime: event.datetime,
                location_name: event.location_name,
                impact_score: edge.impact_score
            }
        """
        
        try:
            cursor = self.db.db.aql.execute(query, bind_vars={'severity': min_severity})
            return list(cursor)
        except Exception as e:
            self.logger.error(f"Error retrieving suppliers affected by weather: {str(e)}")
            return []
    
    def get_weather_impact_summary(self) -> Dict:
        """
        Get a summary of weather impacts on suppliers.
        
        Returns:
            Summary statistics
        """
        query = """
        LET all_impacts = (
            FOR edge IN supplier_affected_by_weather
                LET supplier = DOCUMENT(edge._from)
                LET event = DOCUMENT(edge._to)
                RETURN {
                    supplier_id: supplier.supplier_id,
                    event_type: event.event_type,
                    severity: event.severity,
                    impact_score: edge.impact_score
                }
        )
        
        RETURN {
            total_suppliers_affected: COUNT(UNIQUE(all_impacts[*].supplier_id)),
            total_weather_events: COUNT(UNIQUE(
                FOR edge IN supplier_affected_by_weather
                RETURN DOCUMENT(edge._to).event_id
            )),
            event_type_counts: (
                FOR impact IN all_impacts
                COLLECT event_type = impact.event_type WITH COUNT INTO count
                SORT count DESC
                RETURN {
                    event_type: event_type,
                    count: count
                }
            ),
            severity_counts: (
                FOR impact IN all_impacts
                COLLECT severity = impact.severity WITH COUNT INTO count
                SORT severity DESC
                RETURN {
                    severity: severity,
                    count: count
                }
            ),
            avg_impact_score: AVERAGE(all_impacts[*].impact_score)
        }
        """
        
        try:
            cursor = self.db.db.aql.execute(query)
            results = list(cursor)
            return results[0] if results else {}
        except Exception as e:
            self.logger.error(f"Error retrieving weather impact summary: {str(e)}")
            return {}
    
    def refresh_weather_data(self, min_severity: int = 0) -> Dict:
        """
        Refresh all weather data for suppliers.
        
        Args:
            min_severity: Minimum severity threshold for weather events
            
        Returns:
            Summary of the refresh operation
        """
        start_time = time.time()
        
        try:
            # Update supplier locations
            suppliers_updated = self.update_supplier_locations()
            
            # Collect weather data
            suppliers_processed, events_stored = self.collect_weather_for_suppliers(
                severity_threshold=min_severity
            )
            
            # Get impact summary
            impact_summary = self.get_weather_impact_summary()
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'suppliers_updated': suppliers_updated,
                'suppliers_processed': suppliers_processed,
                'events_stored': events_stored,
                'processing_time_seconds': processing_time,
                'impact_summary': impact_summary
            }
            
        except Exception as e:
            self.logger.error(f"Error refreshing weather data: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': str(e),
                'processing_time_seconds': time.time() - start_time
            }