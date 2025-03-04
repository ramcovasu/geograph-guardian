import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import time
from datetime import datetime, timedelta
import traceback

from src.utils.logging import Logger
from src.weather.weather_data_manager import WeatherDataManager


def render_weather_impact_ui():
    """Render the Weather Impact Analysis UI tab."""
    logger = Logger().get_logger()
    
    st.title("🌦️ Weather Impact Analysis")
    st.markdown("##### Analyze weather impacts on your supply chain")
    
    # Initialize data manager
    try:
        data_manager = WeatherDataManager()
    except Exception as e:
        st.error(f"Error initializing weather data manager: {str(e)}")
        logger.error(f"Error initializing weather data manager: {str(e)}")
        logger.error(traceback.format_exc())
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Controls")
        
        # Data refresh button
        if st.button("🔄 Refresh Weather Data", type="primary"):
            with st.spinner("Refreshing weather data..."):
                try:
                    start_time = time.time()
                    
                    # Update supplier locations first
                    with st.status("Updating supplier locations...") as status:
                        suppliers_updated = data_manager.update_supplier_locations()
                        status.update(label=f"Updated {suppliers_updated} supplier locations", state="complete")
                    
                    # Then collect weather data
                    with st.status("Collecting weather data...") as status:
                        suppliers_processed, events_stored = data_manager.collect_weather_for_suppliers()
                        status.update(label=f"Processed {suppliers_processed} suppliers, stored {events_stored} weather events", state="complete")
                    
                    processing_time = time.time() - start_time
                    
                    st.success(f"Data refresh completed in {processing_time:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"Error refreshing weather data: {str(e)}")
                    logger.error(f"Error in weather data refresh: {str(e)}")
                    logger.error(traceback.format_exc())
        
        # Filter controls
        st.subheader("Filters")
        
        severity_filter = st.slider(
            "Minimum Severity",
            min_value=0,
            max_value=5,
            value=1,
            help="Filter weather events by minimum severity (0-5)"
        )
        
        country_filter = st.text_input(
            "Country Filter",
            value="",
            placeholder="Enter country name",
            help="Filter by country name (leave empty for all)"
        )
        
        # Event type filter
        event_types = [
            "All Types",
            "thunderstorm",
            "rain",
            "snow",
            "clouds",
            "clear",
            "atmosphere",
            "drizzle"
        ]
        
        event_type_filter = st.selectbox(
            "Event Type",
            options=event_types,
            index=0,
            help="Filter by weather event type"
        )
        
        # Apply filters button
        apply_filters = st.button("Apply Filters")
    
    # Main content area
    with col2:
        # Get data based on filters
        if apply_filters or 'weather_data_loaded' not in st.session_state:
            st.session_state.weather_data_loaded = True
            
            # Show a spinner while loading data
            with st.spinner("Loading weather impact data..."):
                try:
                    # Get impact summary
                    impact_summary = data_manager.get_weather_impact_summary()
                    
                    # Get affected suppliers based on filters
                    affected_suppliers = []
                    
                    # Apply filters
                    if country_filter:
                        # Get weather events for specific country
                        weather_events = data_manager.get_weather_for_country(
                            country=country_filter,
                            min_severity=severity_filter
                        )
                    else:
                        # Get all severe weather events
                        weather_events = data_manager.get_severe_weather_events(
                            min_severity=severity_filter
                        )
                    
                    # Apply event type filter
                    if event_type_filter != "All Types":
                        weather_events = [e for e in weather_events if e.get('event_type') == event_type_filter]
                    
                    # Get affected suppliers
                    affected_suppliers = data_manager.get_suppliers_affected_by_weather(
                        min_severity=severity_filter
                    )
                    
                    # Apply country filter to suppliers if specified
                    if country_filter:
                        affected_suppliers = [
                            s for s in affected_suppliers 
                            if country_filter.upper() in s.get('country', '').upper()
                        ]
                    
                    # Apply event type filter to suppliers if specified
                    if event_type_filter != "All Types":
                        affected_suppliers = [
                            s for s in affected_suppliers 
                            if s.get('event_type') == event_type_filter
                        ]
                    
                    # Store data in session state
                    st.session_state.impact_summary = impact_summary
                    st.session_state.weather_events = weather_events
                    st.session_state.affected_suppliers = affected_suppliers
                    
                except Exception as e:
                    st.error(f"Error loading weather impact data: {str(e)}")
                    logger.error(f"Error loading weather impact data: {str(e)}")
                    logger.error(traceback.format_exc())
        
        # Display impact summary
        if 'impact_summary' in st.session_state and st.session_state.impact_summary:
            impact_summary = st.session_state.impact_summary
            
            st.subheader("Supply Chain Weather Impact Summary")
            
            # Create metric columns
            metric_cols = st.columns(3)
            
            with metric_cols[0]:
                suppliers_affected = impact_summary.get('total_suppliers_affected', 0)
                st.metric(
                    label="Suppliers Affected",
                    value=suppliers_affected,
                    help="Total number of suppliers affected by weather events"
                )
            
            with metric_cols[1]:
                events_count = impact_summary.get('total_weather_events', 0)
                st.metric(
                    label="Weather Events",
                    value=events_count,
                    help="Total number of weather events affecting suppliers"
                )
            
            with metric_cols[2]:
                # FIX: Handle None value for avg_impact
                avg_impact = impact_summary.get('avg_impact_score', 0)
                if avg_impact is None:
                    avg_impact_str = "0.0%"
                else:
                    avg_impact_str = f"{avg_impact:.1f}%"
                
                st.metric(
                    label="Avg. Impact Score",
                    value=avg_impact_str,
                    help="Average impact score across all suppliers"
                )
            
            # Display event type distribution if available
            event_type_counts = impact_summary.get('event_type_counts', [])
            if event_type_counts:
                # Convert to DataFrame for visualization
                event_df = pd.DataFrame(event_type_counts)
                
                # Create the chart
                fig = px.pie(
                    event_df, 
                    values='count', 
                    names='event_type',
                    title='Weather Event Types',
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                
                # Improve layout
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    margin=dict(t=30, b=0, l=0, r=0),
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Display affected suppliers map
        if 'weather_events' in st.session_state and st.session_state.weather_events:
            st.subheader("Weather Event Map")
            
            weather_events = st.session_state.weather_events
            
            # Create a DataFrame for the map
            map_data = []
            
            for event in weather_events:
                if 'coordinates' in event and event['coordinates'].get('lat') and event['coordinates'].get('lon'):
                    map_data.append({
                        'lat': event['coordinates']['lat'],
                        'lon': event['coordinates']['lon'],
                        'event_type': event.get('event_type', 'unknown'),
                        'severity': event.get('severity', 0),
                        'description': event.get('description', ''),
                        'location': event.get('location_name', ''),
                        'country': event.get('country_code', ''),
                        'datetime': event.get('datetime', '')
                    })
            
            if map_data:
                map_df = pd.DataFrame(map_data)
                
                # Create the map
                fig = px.scatter_mapbox(
                    map_df,
                    lat="lat",
                    lon="lon",
                    color="event_type",
                    size="severity",
                    size_max=15,
                    zoom=1,
                    hover_name="location",
                    hover_data=["description", "severity", "country", "datetime"],
                    title="Weather Events Affecting Supply Chain",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                
                # Update map layout
                fig.update_layout(
                    mapbox_style="open-street-map",
                    margin=dict(t=0, b=0, l=0, r=0),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No weather events with location data available for map display")
        
        # Display affected suppliers table
        if 'affected_suppliers' in st.session_state:
            affected_suppliers = st.session_state.affected_suppliers
            
            if affected_suppliers:
                st.subheader(f"Affected Suppliers ({len(affected_suppliers)})")
                
                # Convert to DataFrame for display
                suppliers_df = pd.DataFrame(affected_suppliers)
                
                # Format datetime for better display
                if 'datetime' in suppliers_df.columns:
                    suppliers_df['datetime'] = pd.to_datetime(suppliers_df['datetime']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Simplify columns for display
                display_columns = [
                    'supplier_name', 'country', 'region', 'event_type', 
                    'severity', 'description', 'datetime', 'impact_score'
                ]
                
                display_columns = [col for col in display_columns if col in suppliers_df.columns]
                
                # Rename columns for better display
                column_renames = {
                    'supplier_name': 'Supplier',
                    'country': 'Country',
                    'region': 'Region',
                    'event_type': 'Event Type',
                    'severity': 'Severity',
                    'description': 'Description',
                    'datetime': 'Date/Time',
                    'impact_score': 'Impact Score'
                }
                
                suppliers_df = suppliers_df[display_columns].rename(columns=column_renames)
                
                # Sort by impact score (descending)
                if 'Impact Score' in suppliers_df.columns:
                    suppliers_df = suppliers_df.sort_values(by='Impact Score', ascending=False)
                
                # Format impact score as percentage
                if 'Impact Score' in suppliers_df.columns:
                    # FIX: Handle None values in impact_score
                    suppliers_df['Impact Score'] = suppliers_df['Impact Score'].apply(
                        lambda x: f"{x:.1f}%" if x is not None else "0.0%"
                    )
                
                # Display the table
                st.dataframe(suppliers_df, use_container_width=True)
            else:
                st.info("No suppliers found matching the current filters")
        
        # Detailed analysis section
        with st.expander("Detailed Weather Impact Analysis", expanded=False):
            st.markdown("""
            ### Detailed Weather Impact Analysis
            
            This section provides more detailed insights about weather impacts on your supply chain.
            """)
            
            # Add tabs for different analyses
            analysis_tabs = st.tabs(["Event Timeline", "Supplier Risk", "Alternative Suppliers"])
            
            with analysis_tabs[0]:
                st.markdown("### Weather Event Timeline")
                
                if 'weather_events' in st.session_state and st.session_state.weather_events:
                    events = st.session_state.weather_events
                    
                    # Create timeline data
                    timeline_data = []
                    
                    for event in events:
                        # FIX: Better validation of event data
                        if 'timestamp' in event and event['timestamp']:
                            try:
                                timeline_data.append({
                                    'event_type': event.get('event_type', 'unknown'),
                                    'severity': event.get('severity', 0),
                                    'description': event.get('description', ''),
                                    'location': event.get('location_name', ''),
                                    'timestamp': event.get('timestamp', 0),
                                    'datetime': pd.to_datetime(event.get('datetime', '')) if event.get('datetime') else None,
                                    'is_forecast': event.get('is_forecast', False)
                                })
                            except Exception as e:
                                logger.warning(f"Error parsing event for timeline: {str(e)}")
                                # Skip this event and continue
                                continue
                    
                    if timeline_data:
                        timeline_df = pd.DataFrame(timeline_data)
                        
                        # FIX: Handle empty dataframe or missing columns
                        if not timeline_df.empty and 'datetime' in timeline_df.columns and 'severity' in timeline_df.columns:
                            timeline_df = timeline_df.sort_values('timestamp')
                            
                            # Create a timeline chart
                            fig = px.scatter(
                                timeline_df,
                                x='datetime',
                                y='severity',
                                color='event_type',
                                size='severity',
                                hover_name='description',
                                hover_data=['location', 'is_forecast'],
                                title='Weather Event Timeline',
                                labels={'datetime': 'Date/Time', 'severity': 'Severity (1-5)'},
                                color_discrete_sequence=px.colors.qualitative.Plotly
                            )
                            
                            # Improve layout
                            fig.update_layout(
                                xaxis_title='Date/Time',
                                yaxis_title='Severity',
                                height=400
                            )
                            
                            # Add forecast/current indicator
                            current_time = pd.Timestamp.now()
                            fig.add_shape(
                                type="line",
                                x0=current_time,
                                y0=0,
                                x1=current_time,
                                y1=5,
                                line=dict(color="red", width=2, dash="dash"),
                            )
                            
                            fig.add_annotation(
                                x=current_time, 
                                y=5,
                                text="Current Time",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor="red",
                                arrowsize=1,
                                arrowwidth=2
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Insufficient timeline data available")
                    else:
                        st.info("No timeline data available")
                else:
                    st.info("No weather events available for timeline display")
            
            with analysis_tabs[1]:
                st.markdown("### Supplier Weather Risk Analysis")
                
                if 'affected_suppliers' in st.session_state and st.session_state.affected_suppliers:
                    # Create a supplier risk analysis visualization
                    suppliers = st.session_state.affected_suppliers
                    
                    # Group by supplier and calculate max impact
                    supplier_risks = {}
                    
                    for s in suppliers:
                        supplier_id = s.get('supplier_id')
                        supplier_name = s.get('supplier_name', supplier_id)
                        impact_score = s.get('impact_score', 0)
                        
                        # FIX: Handle None values in impact_score
                        if impact_score is None:
                            impact_score = 0
                        
                        if supplier_name not in supplier_risks or impact_score > supplier_risks[supplier_name]['impact']:
                            supplier_risks[supplier_name] = {
                                'impact': impact_score,
                                'country': s.get('country', ''),
                                'event_type': s.get('event_type', ''),
                                'severity': s.get('severity', 0)
                            }
                    
                    # Convert to DataFrame for visualization
                    risk_df = pd.DataFrame([
                        {
                            'supplier': supplier,
                            'impact': data['impact'],
                            'country': data['country'],
                            'event_type': data['event_type'],
                            'severity': data['severity']
                        }
                        for supplier, data in supplier_risks.items()
                    ])
                    
                    # FIX: Handle empty dataframe
                    if not risk_df.empty:
                        # Sort by impact score
                        risk_df = risk_df.sort_values('impact', ascending=False)
                        
                        # Create the chart
                        fig = px.bar(
                            risk_df.head(15),  # Show top 15 suppliers
                            x='supplier',
                            y='impact',
                            color='event_type',
                            hover_data=['country', 'severity'],
                            title='Top Suppliers by Weather Risk Impact',
                            labels={'supplier': 'Supplier', 'impact': 'Impact Score (%)'},
                            color_discrete_sequence=px.colors.qualitative.Plotly
                        )
                        
                        # Improve layout
                        fig.update_layout(
                            xaxis_title='Supplier',
                            yaxis_title='Impact Score (%)',
                            xaxis={'categoryorder':'total descending'},
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No supplier risk data available for visualization")
                else:
                    st.info("No supplier risk data available")
            
            with analysis_tabs[2]:
                st.markdown("### Alternative Suppliers Analysis")
                st.info("This feature will be available in Part 2 of the implementation")
                
                # Placeholder for future development
                st.markdown("""
                In the next phase, this section will provide:
                
                - Identification of alternative suppliers for affected components
                - Comparison of lead times and costs
                - Geographical risk diversification recommendations
                - Automatic mitigation strategy suggestions
                """)