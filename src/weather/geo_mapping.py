import pandas as pd
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import requests

from src.utils.logging import Logger
from src.utils.db_connection import ArangoDB
from src.weather.weather_service import WeatherService


class GeoMapper:
    """Utility for mapping suppliers to geographic coordinates."""
    
    def __init__(self):
        """Initialize the GeoMapper with required dependencies."""
        self.logger = Logger().get_logger()
        self.db = ArangoDB()
        self.weather_service = WeatherService()
        
        # Load country mappings
        self.country_mappings_path = Path(__file__).parent.parent.parent / 'data' / 'reference' / 'country_data.json'
        self.country_mappings = self._load_country_mappings()
    
    def _load_country_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Load country-to-coordinate mappings from file or create if not exists."""
        if not self.country_mappings_path.exists():
            self.logger.info("Country mappings file not found. Creating default data.")
            self._create_default_country_mappings()
        
        try:
            with open(self.country_mappings_path, 'r') as f:
                mappings = json.load(f)
            
            self.logger.info(f"Loaded {len(mappings)} country mappings")
            return mappings
        except Exception as e:
            self.logger.error(f"Error loading country mappings: {str(e)}")
            return {}
    
    def _create_default_country_mappings(self) -> None:
        """Create default country mappings file with basic data."""
        # Create parent directory if it doesn't exist
        self.country_mappings_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Basic dataset of major countries and their approximate coordinates
        default_mappings = {
            "USA": {"name": "United States", "code": "US", "lat": 37.0902, "lon": -95.7129},
            "CHINA": {"name": "China", "code": "CN", "lat": 35.8617, "lon": 104.1954},
            "INDIA": {"name": "India", "code": "IN", "lat": 20.5937, "lon": 78.9629},
            "JAPAN": {"name": "Japan", "code": "JP", "lat": 36.2048, "lon": 138.2529},
            "GERMANY": {"name": "Germany", "code": "DE", "lat": 51.1657, "lon": 10.4515},
            "UK": {"name": "United Kingdom", "code": "GB", "lat": 55.3781, "lon": -3.4360},
            "FRANCE": {"name": "France", "code": "FR", "lat": 46.2276, "lon": 2.2137},
            "ITALY": {"name": "Italy", "code": "IT", "lat": 41.8719, "lon": 12.5674},
            "BRAZIL": {"name": "Brazil", "code": "BR", "lat": -14.2350, "lon": -51.9253},
            "CANADA": {"name": "Canada", "code": "CA", "lat": 56.1304, "lon": -106.3468},
            "AUSTRALIA": {"name": "Australia", "code": "AU", "lat": -25.2744, "lon": 133.7751},
            "RUSSIA": {"name": "Russia", "code": "RU", "lat": 61.5240, "lon": 105.3188},
            "SOUTH KOREA": {"name": "South Korea", "code": "KR", "lat": 35.9078, "lon": 127.7669},
            "SPAIN": {"name": "Spain", "code": "ES", "lat": 40.4637, "lon": -3.7492},
            "MEXICO": {"name": "Mexico", "code": "MX", "lat": 23.6345, "lon": -102.5528},
            "INDONESIA": {"name": "Indonesia", "code": "ID", "lat": -0.7893, "lon": 113.9213},
            "NETHERLANDS": {"name": "Netherlands", "code": "NL", "lat": 52.1326, "lon": 5.2913},
            "SAUDI ARABIA": {"name": "Saudi Arabia", "code": "SA", "lat": 23.8859, "lon": 45.0792},
            "TURKEY": {"name": "Turkey", "code": "TR", "lat": 38.9637, "lon": 35.2433},
            "SWITZERLAND": {"name": "Switzerland", "code": "CH", "lat": 46.8182, "lon": 8.2275},
            "POLAND": {"name": "Poland", "code": "PL", "lat": 51.9194, "lon": 19.1451},
            "THAILAND": {"name": "Thailand", "code": "TH", "lat": 15.8700, "lon": 100.9925},
            "SWEDEN": {"name": "Sweden", "code": "SE", "lat": 60.1282, "lon": 18.6435},
            "BELGIUM": {"name": "Belgium", "code": "BE", "lat": 50.5039, "lon": 4.4699},
            "MALAYSIA": {"name": "Malaysia", "code": "MY", "lat": 4.2105, "lon": 101.9758},
            "SINGAPORE": {"name": "Singapore", "code": "SG", "lat": 1.3521, "lon": 103.8198},
            "VIETNAM": {"name": "Vietnam", "code": "VN", "lat": 14.0583, "lon": 108.2772},
            "ISRAEL": {"name": "Israel", "code": "IL", "lat": 31.0461, "lon": 34.8516}
        }
        
        with open(self.country_mappings_path, 'w') as f:
            json.dump(default_mappings, f, indent=2)
        
        self.logger.info(f"Created default country mappings with {len(default_mappings)} entries")
    
    def get_coordinates_for_country(self, country: str) -> Optional[Tuple[float, float]]:
        """
        Get coordinates for a country.
        
        Args:
            country: Country name or code
            
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        # Normalize input
        country_key = country.strip().upper()
        
        # Check if country is in our mappings
        if country_key in self.country_mappings:
            country_data = self.country_mappings[country_key]
            return country_data['lat'], country_data['lon']
        
        # Try to get coordinates using the weather service
        try:
            coordinates = self.weather_service.get_coordinates(country)
            if coordinates:
                # Add to our mappings for future use
                lat, lon = coordinates
                self.country_mappings[country_key] = {
                    "name": country,
                    "code": "",  # We don't have the code from this lookup
                    "lat": lat,
                    "lon": lon
                }
                
                # Save updated mappings
                with open(self.country_mappings_path, 'w') as f:
                    json.dump(self.country_mappings, f, indent=2)
                
                return coordinates
        except Exception as e:
            self.logger.warning(f"Error getting coordinates for {country}: {str(e)}")
        
        self.logger.warning(f"Could not find coordinates for country: {country}")
        return None
    
    def get_coordinates_for_region(self, country: str, region: str) -> Optional[Tuple[float, float]]:
        """
        Get coordinates for a country-region pair.
        
        Args:
            country: Country name or code
            region: Region name
            
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        # For now, just try to get coordinates for the region name
        # This is a simplification - ideally, we would have a more sophisticated
        # approach for region-level coordinates
        try:
            region_query = f"{region}, {country}"
            coordinates = self.weather_service.get_coordinates(region_query)
            if coordinates:
                return coordinates
        except Exception as e:
            self.logger.warning(f"Error getting coordinates for {region}, {country}: {str(e)}")
        
        # Fall back to country coordinates
        self.logger.info(f"Falling back to country coordinates for {region}, {country}")
        return self.get_coordinates_for_country(country)
    
    def update_supplier_locations(self) -> int:
        """
        Update the supplier_locations collection with geographic coordinates.
        
        Returns:
            Number of suppliers updated
        """
        # Create collection if it doesn't exist
        if not self.db.db.has_collection('supplier_locations'):
            self.db.db.create_collection('supplier_locations')
            self.logger.info("Created supplier_locations collection")
        
        # Get all suppliers
        query = """
        FOR s IN suppliers
        RETURN {
            supplier_id: s.supplier_id,
            supplier_name: s.supplier_name,
            country: s.country,
            region: s.region
        }
        """
        
        try:
            cursor = self.db.db.aql.execute(query)
            suppliers = list(cursor)
            self.logger.info(f"Retrieved {len(suppliers)} suppliers")
            
            updated_count = 0
            
            for supplier in suppliers:
                try:
                    # Check if supplier is a dict or string
                    if not isinstance(supplier, dict):
                        self.logger.warning(f"Supplier is not a dict: {supplier}")
                        continue
                        
                    supplier_id = supplier.get('supplier_id')
                    
                    # Skip if supplier_id is None
                    if not supplier_id:
                        self.logger.warning(f"Missing supplier_id in supplier: {supplier}")
                        continue
                        
                    country = supplier.get('country', '')
                    region = supplier.get('region', '')
                    
                    # Skip if missing required data
                    if not country:
                        self.logger.warning(f"Skipping supplier {supplier_id}: missing country")
                        continue
                    
                    # Determine coordinates
                    coordinates = None
                    if region:
                        coordinates = self.get_coordinates_for_region(country, region)
                        location_precision = 'region'
                    
                    if not coordinates:
                        coordinates = self.get_coordinates_for_country(country)
                        location_precision = 'country'
                    
                    if not coordinates:
                        self.logger.warning(f"Could not determine coordinates for supplier {supplier_id}")
                        continue
                    
                    lat, lon = coordinates
                    
                    # Prepare document with coordinates as a proper JSON object
                    # FIX: Ensure coordinates is a proper dictionary, not a string
                    location_doc = {
                        '_key': supplier_id,
                        'supplier_id': supplier_id,
                        'supplier_name': supplier.get('supplier_name', ''),
                        'country': country,
                        'region': region,
                        'city': '',  # We don't have city data in the current model
                        'coordinates': {
                            'lat': float(lat),  # Ensure these are numeric
                            'lon': float(lon)   # not strings
                        },
                        'location_precision': location_precision,
                        'last_updated': pd.Timestamp.now().isoformat()
                    }
                    
                    # Log the document we're about to insert for debugging
                    self.logger.info(f"Inserting location for supplier {supplier_id}: {json.dumps(location_doc['coordinates'])}")
                    
                    # Insert or update document
                    try:
                        collection = self.db.db.collection('supplier_locations')
                        # Use document method to ensure proper JSON serialization
                        collection.insert(location_doc, overwrite=True)
                        updated_count += 1
                    except Exception as e:
                        self.logger.error(f"Error updating location for supplier {supplier_id}: {str(e)}")
                        
                except Exception as e:
                    supplier_id = supplier.get('supplier_id') if isinstance(supplier, dict) else supplier
                    self.logger.error(f"Error processing supplier {supplier_id}: {str(e)}")
            
            # Verify that coordinates were stored properly
            self._verify_coordinate_storage()
            
            self.logger.info(f"Updated {updated_count} supplier locations")
            return updated_count
            
        except Exception as e:
            self.logger.error(f"Error updating supplier locations: {str(e)}")
            return 0
            
    def _verify_coordinate_storage(self) -> None:
        """Verify that coordinates are stored as proper objects, not strings."""
        query = """
        FOR loc IN supplier_locations
        LIMIT 2
        RETURN {
            supplier_id: loc.supplier_id,
            coords_type: TYPENAME(loc.coordinates),
            coords: loc.coordinates
        }
        """
        
        try:
            cursor = self.db.db.aql.execute(query)
            results = list(cursor)
            
            for result in results:
                self.logger.info(f"Coordinates type check for {result['supplier_id']}: {result['coords_type']}")
                self.logger.info(f"Coordinates value: {result['coords']}")
                
                # If coordinates are stored as strings, raise an error
                if result['coords_type'] == 'string':
                    self.logger.error(f"Coordinates for {result['supplier_id']} stored as string, not object!")
                    
        except Exception as e:
            self.logger.error(f"Error verifying coordinate storage: {str(e)}")
    
    def get_suppliers_by_country(self, country: str) -> List[Dict]:
        """
        Get all suppliers in a specific country.
        
        Args:
            country: Country name
            
        Returns:
            List of supplier documents with location data
        """
        query = """
        FOR loc IN supplier_locations
        FILTER UPPER(loc.country) == UPPER(@country)
        RETURN loc
        """
        
        try:
            cursor = self.db.db.aql.execute(query, bind_vars={'country': country})
            return list(cursor)
        except Exception as e:
            self.logger.error(f"Error getting suppliers for country {country}: {str(e)}")
            return []