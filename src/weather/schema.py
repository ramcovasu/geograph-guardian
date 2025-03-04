"""
Weather Database Schema module for GeoGraph Guardian.
This file handles the creation and management of weather-related collections and indexes.
"""

from src.utils.db_connection import ArangoDB
from src.utils.logging import Logger
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

class WeatherDatabaseSchema:
    """Creates and manages weather-related database schema extensions."""
    
    def __init__(self):
        """Initialize the schema manager with database connection."""
        self.logger = Logger().get_logger()
        self.db = ArangoDB()
    
    def setup_weather_collections(self) -> bool:
        """
        Set up all weather-related collections.
        
        Returns:
            Success status
        """
        try:
            # Create vertex collections
            self._create_collection('weather_events')
            self._create_collection('supplier_locations')
            self._create_collection('weather_impact_assessments')
            
            # Create edge collection
            self._create_collection('supplier_affected_by_weather', edge=True)
            
            # Create indexes
            self._create_indexes()
            
            # Update the graph definition to include the new collections
            self._extend_graph()
            
            self.logger.info("Successfully set up weather collections")
            return True
        except Exception as e:
            self.logger.error(f"Error setting up weather collections: {str(e)}")
            return False
    
    def _create_collection(self, name: str, edge: bool = False) -> None:
        """Create a collection if it doesn't exist."""
        try:
            if not self.db.db.has_collection(name):
                self.db.db.create_collection(name, edge=edge)
                self.logger.info(f"Created collection: {name}")
            else:
                self.logger.info(f"Collection already exists: {name}")
        except Exception as e:
            self.logger.error(f"Error creating collection {name}: {str(e)}")
            raise
    
    def _create_indexes(self) -> None:
        """Create indexes for weather collections."""
        try:
            # Indexes for weather_events
            weather_events = self.db.db.collection('weather_events')
            
            # Check if indexes already exist before creating
            existing_indexes = [idx['fields'] for idx in weather_events.indexes()]
            
            if ['event_type'] not in existing_indexes:
                weather_events.add_hash_index(['event_type'], unique=False)
                self.logger.info("Created index on weather_events.event_type")
                
            if ['severity'] not in existing_indexes:
                weather_events.add_hash_index(['severity'], unique=False)
                self.logger.info("Created index on weather_events.severity")
                
            if ['timestamp'] not in existing_indexes:
                weather_events.add_skiplist_index(['timestamp'], unique=False)
                self.logger.info("Created index on weather_events.timestamp")
                
            if ['country_code'] not in existing_indexes:
                weather_events.add_hash_index(['country_code'], unique=False)
                self.logger.info("Created index on weather_events.country_code")
                
            # Geo-spatial index on coordinates
            if ['coordinates.lat', 'coordinates.lon'] not in existing_indexes:
                weather_events.add_geo_index(['coordinates.lat', 'coordinates.lon'])
                self.logger.info("Created geo index on weather_events.coordinates")
            
            # Indexes for supplier_locations
            supplier_locations = self.db.db.collection('supplier_locations')
            existing_indexes = [idx['fields'] for idx in supplier_locations.indexes()]
            
            if ['country'] not in existing_indexes:
                supplier_locations.add_hash_index(['country'], unique=False)
                self.logger.info("Created index on supplier_locations.country")
                
            if ['region'] not in existing_indexes:
                supplier_locations.add_hash_index(['region'], unique=False)
                self.logger.info("Created index on supplier_locations.region")
                
            # Geo-spatial index on coordinates
            if ['coordinates.lat', 'coordinates.lon'] not in existing_indexes:
                supplier_locations.add_geo_index(['coordinates.lat', 'coordinates.lon'])
                self.logger.info("Created geo index on supplier_locations.coordinates")
            
            # Indexes for weather_impact_assessments
            impact_assessments = self.db.db.collection('weather_impact_assessments')
            existing_indexes = [idx['fields'] for idx in impact_assessments.indexes()]
            
            if ['supplier_id'] not in existing_indexes:
                impact_assessments.add_hash_index(['supplier_id'], unique=False)
                self.logger.info("Created index on weather_impact_assessments.supplier_id")
                
            if ['impact_score'] not in existing_indexes:
                impact_assessments.add_skiplist_index(['impact_score'], unique=False)
                self.logger.info("Created index on weather_impact_assessments.impact_score")
                
            if ['assessment_date'] not in existing_indexes:
                impact_assessments.add_skiplist_index(['assessment_date'], unique=False)
                self.logger.info("Created index on weather_impact_assessments.assessment_date")
            
        except Exception as e:
            self.logger.error(f"Error creating indexes: {str(e)}")
            raise
    
    def _extend_graph(self) -> None:
        """Extend the supply chain graph with weather-related edge definitions."""
        try:
            # Get the graph
            graph_name = 'supplychain'
            if not self.db.db.has_graph(graph_name):
                self.logger.warning(f"Graph {graph_name} does not exist, creating it")
                graph = self.db.db.create_graph(graph_name)
            else:
                graph = self.db.db.graph(graph_name)
            
            # Add edge definition for supplier affected by weather
            edge_def = {
                'edge_collection': 'supplier_affected_by_weather',
                'from_vertex_collections': ['suppliers'],
                'to_vertex_collections': ['weather_events']
            }
            
            # Check if edge definition already exists
            existing_edge_defs = graph.edge_definitions()
            edge_exists = False
            
            for existing_def in existing_edge_defs:
                if existing_def['edge_collection'] == edge_def['edge_collection']:
                    edge_exists = True
                    break
            
            if not edge_exists:
                graph.create_edge_definition(
                    edge_collection=edge_def['edge_collection'],
                    from_vertex_collections=edge_def['from_vertex_collections'],
                    to_vertex_collections=edge_def['to_vertex_collections']
                )
                self.logger.info(f"Added edge definition {edge_def['edge_collection']} to graph {graph_name}")
            else:
                self.logger.info(f"Edge definition {edge_def['edge_collection']} already exists in graph {graph_name}")
            
        except Exception as e:
            self.logger.error(f"Error extending graph: {str(e)}")
            raise
            
    def clean_weather_data(self) -> bool:
        """
        Clean all weather-related collections (for testing/reset purposes).
        
        Returns:
            Success status
        """
        try:
            collections = [
                'weather_events',
                'supplier_locations',
                'weather_impact_assessments',
                'supplier_affected_by_weather'
            ]
            
            for collection_name in collections:
                if self.db.db.has_collection(collection_name):
                    collection = self.db.db.collection(collection_name)
                    collection.truncate()
                    self.logger.info(f"Truncated collection: {collection_name}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning weather data: {str(e)}")
            return False