import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import traceback
import time

from src.graph_analytics import GraphAnalytics, GraphDataConverter, GraphVisualizer
from src.utils.logging import Logger

def render_graph_analytics_ui():
    """Render the Graph Analytics UI tab."""
    logger = Logger().get_logger()
    
    st.title("🧠 Graph Analytics")
    st.markdown("##### GPU-accelerated graph analysis for supply chain networks")
    
    # Initialize objects
    try:
        analytics = GraphAnalytics()
        converter = GraphDataConverter()
        visualizer = GraphVisualizer()
        
        # Check if CUDA is available
        cuda_available = analytics._check_cuda_availability()
        
        if not cuda_available:
            st.warning("⚠️ GPU acceleration not available. Some features may be limited or slower.")
    except Exception as e:
        st.error(f"Error initializing graph analytics components: {str(e)}")
        logger.error(f"Error initializing graph analytics components: {str(e)}")
        logger.error(traceback.format_exc())
        return
    
    # Create two columns for the interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📊 Analysis Options")
        
        # Graph source selection
        graph_source = st.selectbox(
            "Select Graph Source",
            ["Supplier-Part Network", "Part Dependencies", "Supplier Risk Network", "Combined Supply Chain", "Custom Query"]
        )
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Community Detection", "Centrality Analysis", "Shortest Path Analysis", "Network Visualization"]
        )
        
        # Additional options based on analysis type
        if analysis_type == "Centrality Analysis":
            # Get vertex count for sample size limit
            # We don't have vertex count yet, will be set after graph is created
            max_sample = 500  # Default max
            
            centrality_method = st.selectbox(
                "Centrality Method",
                ["betweenness", "eigenvector", "pagerank"]
            )
            
            sample_size = st.slider(
                "Sample Size (for large graphs)",
                min_value=10,
                max_value=max_sample,
                value=100,
                step=10
            )
        
        if analysis_type == "Shortest Path Analysis":
            # Dynamic source vertex selection would go here
            # For now, use a text input
            source_vertex = st.text_input(
                "Source Vertex ID",
                value="S001"  # Default value
            )
        
        if graph_source == "Custom Query":
            custom_query = st.text_area(
                "Enter AQL Query (must return source and target columns)",
                height=150,
                value="""
FOR sp IN supplier_provides_part
    LET supplier = DOCUMENT(sp._from)
    LET part = DOCUMENT(sp._to)
    RETURN {
        source: supplier.supplier_id,
        target: part.part_id
    }
                """
            )
        
        # Run analysis button
        run_button = st.button("Run Analysis", type="primary")
    
    # Main content area
    with col2:
        st.markdown("### 📈 Analysis Results")
        
        if run_button:
            with st.spinner("Running graph analysis..."):
                try:
                    # Get graph data based on selection
                    if graph_source == "Supplier-Part Network":
                        graph_df = converter.prepare_supplier_part_graph()
                    elif graph_source == "Part Dependencies":
                        graph_df = converter.prepare_part_dependency_graph()
                    elif graph_source == "Supplier Risk Network":
                        graph_df = converter.prepare_supplier_risk_graph()
                    elif graph_source == "Combined Supply Chain":
                        graph_df = converter.prepare_combined_supply_chain_graph()
                    elif graph_source == "Custom Query":
                        graph_df = converter.extract_graph_from_query(custom_query)
                    else:
                        raise ValueError(f"Unknown graph source: {graph_source}")
                    
                    if graph_df is None or len(graph_df) == 0:
                        st.error(f"No data available for the selected graph source: {graph_source}")
                        return
                    
                    # Display dataset info
                    st.markdown(f"**Dataset:** {len(graph_df)} edges")
                    
                    # Convert to cuGraph
                    start_time = time.time()
                    G = analytics.convert_to_cugraph(graph_df)
                    
                    if G is None:
                        st.error("Failed to convert data to cuGraph format")
                        return
                    
                    # Get graph stats
                    stats = analytics.get_graph_stats(G)
                    st.markdown(f"**Graph Stats:** {stats['num_vertices']} nodes, {stats['num_edges']} edges, Density: {stats['density']:.4f}")
                    
                    # Run analysis based on selection
                    if analysis_type == "Community Detection":
                        result_df = analytics.run_community_detection(G)
                        
                        if result_df is None:
                            st.error("Community detection analysis failed")
                            return
                        
                        # Get the figure from visualize_communities
                        fig = visualizer.visualize_communities(
                            result_df,
                            edge_df=graph_df,  # Pass original edge data for visualization
                            graph_type=graph_source
                        )
                        
                        # Separately get a narrative for the community detection results
                        # Prepare communities data
                        communities = {}
                        for _, row in result_df.iterrows():
                            community = row['partition']
                            vertex = row['vertex']
                            if community not in communities:
                                communities[community] = []
                            communities[community].append(vertex)
                        
                        # Get modularity score
                        modularity = result_df.attrs.get('modularity', 0)
                        
                        # Prepare community data for narrative generation
                        community_data = {
                            'num_communities': len(communities),
                            'community_sizes': {comm: len(nodes) for comm, nodes in communities.items()},
                            'modularity': modularity
                        }
                        
                        # Generate narrative using LLM
                        narrative = visualizer.get_llm_narrative(community_data, graph_source)
                        
                        # Display results summary
                        num_communities = result_df['partition'].nunique()
                        st.markdown(f"**Found {num_communities} communities**")
                        
                        # Display the figure with a unique key
                        if isinstance(fig, go.Figure):
                            st.plotly_chart(fig, use_container_width=True, key="community_detection_chart")
                        else:
                            st.error(f"Invalid figure object type: {type(fig)}")
                            
                        # Display the narrative explanation
                        st.markdown("### 📝 Analysis Explanation")
                        st.markdown(f"""
                        <div class="explanation-wrapper">
                            {narrative}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show sample of raw data
                        with st.expander("Raw Data"):
                            st.dataframe(result_df.head(20))
                    
                    elif analysis_type == "Centrality Analysis":
                        # Get vertex count for sample size limit
                        vertex_count = stats['num_vertices']
                        
                        # Adjust max slider value based on vertex count
                        max_sample = min(500, vertex_count)
                        sample_size = st.slider(
                            "Sample Size (for large graphs)",
                            min_value=10,
                            max_value=max_sample,
                            value=min(100, max_sample),
                            step=10,
                            key="sample_size_updated"
                        )
                        
                        # Show warning if dataset is small
                        if vertex_count < 50:
                            st.warning(f"Small graph detected ({vertex_count} vertices). Some centrality metrics may have limited value.")
                        
                        # Show progress to provide feedback during potentially long-running operations
                        with st.status("Running centrality analysis...") as status:
                            status.update(label="Calculating centrality scores...")
                            
                            result_df = analytics.run_centrality_analysis(
                                G, 
                                method=centrality_method,
                                k=sample_size
                            )
                            
                            if result_df is None:
                                st.error("Centrality analysis failed")
                                return
                            
                            # Check for valid columns in the result DataFrame
                            if len(result_df.columns) < 2:
                                st.error(f"Invalid result DataFrame: expected at least 2 columns, got {len(result_df.columns)}")
                                st.write("Columns:", result_df.columns.tolist())
                                return
                                
                            # Determine vertex column and centrality column
                            vertex_col = 'vertex'  # Default name
                            # Find the centrality column (should be the non-vertex column)
                            centrality_cols = [col for col in result_df.columns if col != vertex_col]
                            if not centrality_cols:
                                st.error("No centrality measure column found in results")
                                return
                            centrality_col = centrality_cols[0]
                            
                            # Check if we're using a fallback method
                            actual_method = result_df.attrs.get('method', centrality_method)
                            if actual_method != centrality_method:
                                st.warning(f"⚠️ Using {actual_method} as a fallback because {centrality_method} failed")
                            
                            status.update(label="Generating visualization...")
                            
                            # Visualize results
                            fig = visualizer.visualize_centrality(result_df, vertex_col=vertex_col, centrality_col=centrality_col)
                            
                            # Generate narrative explanation
                            status.update(label="Generating narrative explanation...")
                            narrative = visualizer.get_centrality_narrative(
                                data=result_df,
                                method=actual_method,
                                graph_type=graph_source
                            )
                            
                            status.update(label="Analysis complete", state="complete")
                        
                        # Display the heading for results
                        st.subheader(f"📊 {actual_method.title()} Centrality Results")
                        
                        # Display the figure with a unique key
                        if isinstance(fig, go.Figure):
                            st.plotly_chart(fig, use_container_width=True, key=f"{actual_method}_centrality_chart")
                        else:
                            st.error(f"Invalid figure object type: {type(fig)}")
                        
                        # Display the narrative explanation
                        st.markdown("### 📝 Analysis Explanation")
                        st.markdown(f"""
                        <div class="explanation-wrapper">
                            {narrative}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show node importance summary
                        if len(result_df) > 0:
                            try:
                                # Use the centrality column for nlargest, not the vertex column
                                top_5 = result_df.nlargest(min(5, len(result_df)), centrality_col)
                                
                                with st.expander("Most Important Nodes Summary"):
                                    importance_cols = st.columns(min(5, len(top_5)))
                                    for i, (_, row) in enumerate(top_5.iterrows()):
                                        with importance_cols[i]:
                                            st.metric(
                                                label=f"#{i+1}",
                                                value=str(row[vertex_col]),
                                                delta=f"{row[centrality_col]:.3f}"
                                            )
                            except Exception as e:
                                st.warning(f"Could not display node importance summary: {str(e)}")
                        
                        # Show sample of raw data
                        with st.expander("Raw Data"):
                            try:
                                # Sort by centrality column if it's numeric
                                if pd.api.types.is_numeric_dtype(result_df[centrality_col]):
                                    st.dataframe(result_df.sort_values(centrality_col, ascending=False).head(20))
                                else:
                                    st.dataframe(result_df.head(20))
                            except Exception as e:
                                st.warning(f"Could not sort data: {str(e)}")
                                st.dataframe(result_df.head(20))
                    
                    elif analysis_type == "Shortest Path Analysis":
                        if not source_vertex:
                            st.error("Source vertex must be specified")
                            return
                        
                        result_df = analytics.run_shortest_path(source_vertex, G)
                        
                        if result_df is None:
                            st.error(f"Shortest path analysis from {source_vertex} failed")
                            return
                        
                        # Visualize results
                        fig = visualizer.visualize_shortest_paths(result_df)
                        
                        # Display the figure with a unique key
                        if isinstance(fig, go.Figure):
                            st.plotly_chart(fig, use_container_width=True, key="shortest_path_chart")
                        else:
                            st.error(f"Invalid figure object type: {type(fig)}")
                        
                        # Show sample of raw data
                        with st.expander("Raw Data"):
                            st.dataframe(result_df.sort_values('distance').head(20))
                    
                    elif analysis_type == "Network Visualization":
                        # Network visualization using networkx
                        fig = visualizer.create_network_graph(graph_df)
                        
                        # Display the figure with a unique key
                        if isinstance(fig, go.Figure):
                            st.plotly_chart(fig, use_container_width=True, key="network_visualization_chart")
                        else:
                            st.error(f"Invalid figure object type: {type(fig)}")
                    
                    # Show processing time
                    processing_time = time.time() - start_time
                    st.markdown(f"*Processing time: {processing_time:.2f} seconds*")
                    
                except Exception as e:
                    st.error(f"Error running analysis: {str(e)}")
                    logger.error(f"Error in graph analytics: {str(e)}")
                    logger.error(traceback.format_exc())
    
    # Additional information
    with st.expander("About Graph Analytics"):
        st.markdown("""
        ### GPU-Accelerated Graph Analytics
        
        This module uses NVIDIA cuGraph to perform high-performance graph analytics on your supply chain network:
        
        - **Community Detection**: Identifies clusters or communities in your supply chain, revealing hidden patterns of interconnected suppliers and parts
        - **Centrality Analysis**: Finds the most important/central nodes in your network, highlighting critical suppliers or parts that may represent single points of failure
        - **Shortest Path Analysis**: Computes optimal paths between nodes, showing dependencies and potential alternative routes
        - **Network Visualization**: Visualizes the structure of your supply chain network to reveal patterns not visible in traditional reports
        
        These analyses can help identify critical suppliers, component clusters, and potential bottlenecks in your supply chain.
        
        The GPU acceleration provided by NVIDIA cuGraph enables analyzing complex supply chains with thousands of components in seconds, providing insights that would take minutes or hours with traditional methods.
        """)