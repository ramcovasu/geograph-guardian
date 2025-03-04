#!/usr/bin/env python
"""
Script to set up the weather database schema and initialize the system.
This script incorporates the weather schema extensions into the existing supply chain graph.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import Logger
from src.weather.schema import WeatherDatabaseSchema
from src.weather.weather_data_manager import WeatherDataManager
from src.utils.db_connection import ArangoDB

def main():
    """Main function to set up weather database schema."""
    logger = Logger().get_logger()
    
    parser = argparse.ArgumentParser(description='Set up weather database schema')
    parser.add_argument('--reset', action='store_true', help='Reset existing weather data')
    parser.add_argument('--populate', action='store_true', help='Populate with initial data')
    args = parser.parse_args()
    
    try:
        logger.info("Starting weather database setup")
        
        # Check database connection
        db = ArangoDB()
        db.db.collections()
        logger.info("Database connection successful")
        
        # Set up schema
        schema = WeatherDatabaseSchema()
        
        # Reset data if requested
        if args.reset:
            logger.info("Resetting weather data...")
            schema.clean_weather_data()
        
        # Create collections and indexes
        logger.info("Setting up weather collections...")
        success = schema.setup_weather_collections()
        
        if not success:
            logger.error("Failed to set up weather collections")
            return 1
        
        # Populate with initial data if requested
        if args.populate:
            logger.info("Populating initial weather data...")
            data_manager = WeatherDataManager()
            
            # Update supplier locations
            suppliers_updated = data_manager.update_supplier_locations()
            logger.info(f"Updated {suppliers_updated} supplier locations")
            
            # Collect initial weather data
            suppliers_processed, events_stored = data_manager.collect_weather_for_suppliers()
            logger.info(f"Processed {suppliers_processed} suppliers, stored {events_stored} weather events")
            
            # Get a summary of impacts
            impact_summary = data_manager.get_weather_impact_summary()
            logger.info(f"Weather impact summary: {impact_summary}")
        
        logger.info("Weather database setup completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in weather database setup: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())