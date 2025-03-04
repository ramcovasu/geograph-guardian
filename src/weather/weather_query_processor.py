"""
Weather Query Processor Extension for GeoGraph Guardian.

This file demonstrates how the query processor will be extended in Part 2 to handle weather-related queries.
This is a preview and not part of the Part 1 implementation.
"""

from typing import Dict, List, Any, Optional
import re

class WeatherQueryProcessor:
    """Extension for the query processor to handle weather-related queries."""
    
    def __init__(self, query_processor):
        """
        Initialize with a reference to the main query processor.
        
        Args:
            query_processor: The main QueryProcessor instance
        """
        self.query_processor = query_processor
        
        # Add weather-related schema context
        self._extend_schema_context()
        
        # Add weather-related query templates
        self._add_weather_query_templates()
    
    def _extend_schema_context(self):
        """Extend the schema context with weather-related collections."""
        weather_schema_context = """
        Additional Collections:
        
        - weather_events: event_id, event_type, severity, description, timestamp, location_name, country_code, coordinates
        - supplier_locations: supplier_id, country, region, coordinates
        - weather_impact_assessments: assessment_id, weather_event_id, supplier_id, impact_score
        
        Additional Relationships (Graph Edges):
        
        - supplier_affected_by_weather: suppliers → weather_events
        
        Example Weather Queries:
        
        1. Find suppliers affected by severe weather:
        FOR edge IN supplier_affected_by_weather
            LET supplier = DOCUMENT(edge._from)
            LET event = DOCUMENT(edge._to)
            FILTER event.severity >= 3
            RETURN {
                supplier_name: supplier.supplier_name,
                country: supplier.country,
                event_type: event.event_type,
                severity: event.severity,
                description: event.description,
                impact_score: edge.impact_score
            }
        
        2. Find weather events by country:
        FOR event IN weather_events
            FILTER UPPER(event.country_code) == "US"
            SORT event.severity DESC
            LIMIT 10
            RETURN event
        
        3. Find alternative suppliers for parts affected by weather:
        FOR edge IN supplier_affected_by_weather
            FILTER edge.impact_score > 50
            LET affected_supplier = DOCUMENT(edge._from)
            LET weather_event = DOCUMENT(edge._to)
            
            LET affected_parts = (
                FOR sp IN supplier_provides_part
                FILTER sp._from == affected_supplier._id
                LET part = DOCUMENT(sp._to)
                RETURN part
            )
            
            LET alternative_suppliers = (
                FOR part IN affected_parts
                FOR sp IN supplier_provides_part
                FILTER sp._to == part._id
                FILTER sp._from != affected_supplier._id
                LET supplier = DOCUMENT(sp._from)
                
                // Check if alternative supplier is not affected by same weather
                LET is_affected = (
                    FOR e IN supplier_affected_by_weather
                    FILTER e._from == supplier._id
                    FILTER e._to == weather_event._id
                    RETURN 1
                )
                
                FILTER LENGTH(is_affected) == 0
                
                RETURN {
                    supplier_id: supplier.supplier_id,
                    supplier_name: supplier.supplier_name,
                    part_id: part.part_id,
                    part_name: part.part_name
                }
            )
            
            RETURN {
                affected_supplier: affected_supplier.supplier_name,
                weather_event: weather_event.event_type,
                severity: weather_event.severity,
                alternative_suppliers: alternative_suppliers
            }
        """
        
        # In Part 2, we would add this to the query processor's schema context
        # self.query_processor.schema_context += weather_schema_context
    
    def _add_weather_query_templates(self):
        """Add weather-related query templates to the query processor."""
        weather_templates = {
            "weather_events_by_country": """
                FOR event IN weather_events
                FILTER UPPER(event.country_code) == UPPER({country})
                FILTER event.severity >= {min_severity}
                SORT event.severity DESC, event.timestamp DESC
                LIMIT {limit}
                RETURN event
            """.strip(),
            
            "suppliers_affected_by_weather": """
                FOR edge IN supplier_affected_by_weather
                LET supplier = DOCUMENT(edge._from)
                LET event = DOCUMENT(edge._to)
                FILTER event.severity >= {min_severity}
                SORT edge.impact_score DESC
                LIMIT {limit}
                RETURN {
                    supplier_id: supplier.supplier_id,
                    supplier_name: supplier.supplier_name,
                    country: supplier.country,
                    region: supplier.region,
                    event_type: event.event_type,
                    severity: event.severity,
                    description: event.description,
                    impact_score: edge.impact_score
                }
            """.strip(),
            
            "alternative_suppliers_for_weather_affected": """
                LET affected_suppliers = (
                    FOR edge IN supplier_affected_by_weather
                    FILTER edge.impact_score > {impact_threshold}
                    RETURN DOCUMENT(edge._from)
                )
                
                LET affected_parts = (
                    FOR supplier IN affected_suppliers
                    FOR sp IN supplier_provides_part
                    FILTER sp._from == supplier._id
                    RETURN DOCUMENT(sp._to)
                )
                
                FOR part IN affected_parts
                LET alternatives = (
                    FOR sp IN supplier_provides_part
                    FILTER sp._to == part._id
                    LET supplier = DOCUMENT(sp._from)
                    
                    // Check if alternative is not in affected list
                    FILTER supplier NOT IN affected_suppliers
                    
                    RETURN {
                        supplier_id: supplier.supplier_id,
                        supplier_name: supplier.supplier_name,
                        lead_time: sp.lead_time_days,
                        unit_cost: sp.unit_cost
                    }
                )
                
                FILTER LENGTH(alternatives) > 0
                
                RETURN {
                    part_id: part.part_id,
                    part_name: part.part_name,
                    criticality: part.criticality_level,
                    alternative_suppliers: alternatives
                }
            """.strip()
        }
        
        # In Part 2, we would add these templates to the query processor
        # self.query_processor._query_templates.update(weather_templates)
    
    def detect_weather_query(self, query_text: str) -> bool:
        """
        Detect if a query is related to weather impacts.
        
        Args:
            query_text: Natural language query text
            
        Returns:
            True if query is weather-related, False otherwise
        """
        weather_keywords = [
            "weather", "storm", "hurricane", "typhoon", "flood", "rain",
            "snow", "tornado", "cyclone", "forecast", "precipitation",
            "thunderstorm", "blizzard", "temperature"
        ]
        
        impact_keywords = [
            "impact", "affect", "disrupt", "delay", "risk",
            "alternative", "backup", "contingency", "affected"
        ]
        
        # Check for weather keywords
        has_weather_term = any(re.search(r'\b' + kw + r'\b', query_text, re.IGNORECASE) for kw in weather_keywords)
        
        # Check for impact keywords
        has_impact_term = any(re.search(r'\b' + kw + r'\b', query_text, re.IGNORECASE) for kw in impact_keywords)
        
        # Check for supplier or part terms
        has_supply_term = re.search(r'\b(supplier|part|component|inventory)\b', query_text, re.IGNORECASE) is not None
        
        # Consider it a weather query if it has a weather term AND either an impact term or supply term
        return has_weather_term and (has_impact_term or has_supply_term)

# Example usage (to be implemented in Part 2):
"""
# Extend QueryProcessor in __init__ method:

def __init__(self):
    # ... existing initialization ...
    
    # Add weather query processor extension
    self.weather_processor = WeatherQueryProcessor(self)

# Then in generate_aql method:

def generate_aql(self, question: str) -> str:
    # Check if it's a weather-related query
    if self.weather_processor.detect_weather_query(question):
        # Use weather-specific prompt template
        # ...
    
    # Continue with regular query processing
    # ...
"""