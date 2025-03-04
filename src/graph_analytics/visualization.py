import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple, Union
from src.utils.logging import Logger
import networkx as nx
import traceback  

class GraphVisualizer:
    """Specialized visualizations for graph analytics results."""
    
    def __init__(self):
        """Initialize the GraphVisualizer class."""
        self.logger = Logger().get_logger()
        self.default_height = 600
        self.default_width = 800
        self.color_scale = 'Viridis'
        self.community_colors = px.colors.qualitative.Bold
    
    def _identify_key_entities(self, communities: Dict[int, List], graph: nx.Graph) -> Dict[int, List]:
        """
        Identify key entities for each community.
        
        Args:
            communities: Dictionary mapping community IDs to list of nodes
            graph: NetworkX graph with node attributes
            
        Returns:
            Dictionary of key entities for each community
        """
        key_entities = {}
        
        for community_id, nodes in communities.items():
            # Collect node attributes
            node_attrs = []
            for node in nodes:
                attrs = graph.nodes.get(node, {})
                node_attrs.append({
                    'id': node,
                    'name': attrs.get('name', node),
                    'type': attrs.get('type', 'unknown'),
                    'category': attrs.get('category', ''),
                    'rating': attrs.get('rating', ''),
                    'criticality': attrs.get('criticality', '')
                })
            
            # Sort and select key entities
            # Prioritize by type, then by specific criteria
            key_entities[community_id] = sorted(
                node_attrs, 
                key=lambda x: (
                    0 if x['type'] == 'supplier' else 
                    1 if x['type'] == 'part' else 2,
                    -float(x.get('rating', 0) or 0),
                    -float(x.get('criticality', 0) or 0)
                ), 
                reverse=True
            )[:5]  # Take top 5 key entities
        
        return key_entities

    # Modify the visualize_communities method in src/visualization.py
    def visualize_communities(self, data: pd.DataFrame, 
                            vertex_col: str = 'vertex', 
                            partition_col: str = 'partition',
                            edge_df: Optional[pd.DataFrame] = None,
                            source_col: str = 'source',
                            target_col: str = 'target',
                            graph_type: str = "Supply Chain") -> go.Figure:
        """
        Create a visualization for community detection results.
        
        Args:
            data: DataFrame with vertex and partition columns
            vertex_col: Column name for vertex IDs
            partition_col: Column name for community assignments
            edge_df: Optional DataFrame with edge data
            source_col: Column name for source vertices in edge_df
            target_col: Column name for target vertices in edge_df
            graph_type: Type of graph for context-specific narrative
            
        Returns:
            Plotly figure for community visualization
        """
        try:
            # Validate input data
            if data is None or len(data) == 0:
                return self.create_error_chart("No community data available")

            if vertex_col not in data.columns or partition_col not in data.columns:
                return self.create_error_chart(f"Missing required columns: {vertex_col} or {partition_col}")

            # Create NetworkX graph for layout
            import networkx as nx
            G = nx.Graph()
            
            # Add nodes with community information
            for _, row in data.iterrows():
                vertex = row[vertex_col]
                community = row[partition_col]
                
                # Get node metadata if available
                node_attrs = {'community': community}
                if edge_df is not None and hasattr(edge_df, 'attrs') and 'node_metadata' in edge_df.attrs:
                    node_metadata = edge_df.attrs['node_metadata']
                    if vertex in node_metadata:
                        node_attrs.update(node_metadata[vertex])
                
                G.add_node(vertex, **node_attrs)
            
            # Add edges if provided
            if edge_df is not None and source_col in edge_df.columns and target_col in edge_df.columns:
                for _, row in edge_df.iterrows():
                    source = row[source_col]
                    target = row[target_col]
                    # Only add edge if both nodes exist
                    if source in G.nodes and target in G.nodes:
                        G.add_edge(source, target)
            
            # Calculate layout
            pos = nx.spring_layout(G, seed=42)
            
            # Prepare communities
            communities = {}
            for node, attrs in G.nodes(data=True):
                community = attrs.get('community', -1)
                if community not in communities:
                    communities[community] = []
                communities[community].append(node)
            
            # Identify key entities
            try:
                community_key_entities = self._identify_key_entities(communities, G)
            except Exception as key_entity_error:
                self.logger.warning(f"Could not identify key entities: {str(key_entity_error)}")
                community_key_entities = {}
            
            # Prepare node traces
            node_traces = []
            for i, (community, nodes) in enumerate(communities.items()):
                x = [pos[node][0] for node in nodes]
                y = [pos[node][1] for node in nodes]
                
                # Assign color from community color palette
                color = self.community_colors[i % len(self.community_colors)]
                
                # Prepare hover text
                hover_text = []
                node_sizes = []
                for node in nodes:
                    attrs = G.nodes[node]
                    name = attrs.get('name', node)
                    node_type = attrs.get('type', 'Unknown')
                    
                    hover_text.append(f"ID: {node}<br>Name: {name}<br>Type: {node_type}<br>Community: {community}")
                    
                    # Set node size based on type
                    if node_type == 'supplier':
                        node_sizes.append(20)
                    elif node_type == 'part':
                        node_sizes.append(15)
                    else:
                        node_sizes.append(10)
                
                # Create node trace
                node_trace = go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(
                        size=node_sizes,
                        color=color,
                        line=dict(width=1, color='black')
                    ),
                    text=hover_text,
                    hoverinfo='text',
                    name=f"Community {community}"
                )
                node_traces.append(node_trace)
            
            # Create edge trace
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            
            # Get modularity score
            modularity = data.attrs.get('modularity', 0)
            
            # Create figure
            fig = go.Figure(data=[edge_trace] + node_traces)
            fig.update_layout(
                title=f"Community Detection Results (Modularity: {modularity:.4f})",
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=self.default_height,
                width=self.default_width
            )
            
            # Generate narrative
            narrative = self.generate_community_narrative(data, communities, graph_type)
            
            # Return just the figure object for compatibility with Plotly
            return fig
            
        except Exception as e:
            # Create an error figure if anything goes wrong
            error_fig = self.create_error_chart(f"Error visualizing communities: {str(e)}")
            self.logger.error(f"Error visualizing communities: {str(e)}")
            self.logger.error(traceback.format_exc())
            return error_fig
        
        
    def get_llm_narrative(self, community_data: Dict, graph_type: str) -> str:
        """
        Get a narrative explanation for community detection from the LLM.
        
        Args:
            community_data: Dictionary containing community analysis results
            graph_type: Type of graph for context
            
        Returns:
            Narrative explanation from LLM
        """
        try:
            from src.llm.query_processor import QueryProcessor
            
            # Initialize query processor for LLM access
            processor = QueryProcessor()
            
            # Format information about communities for the prompt
            num_communities = community_data.get('num_communities', 0)
            community_sizes = community_data.get('community_sizes', {})
            node_types = community_data.get('node_types', {})
            modularity = community_data.get('modularity', 0)
            key_entities = community_data.get('key_entities', {})
            
            # Create detailed information about each community
            community_info = []
            for comm_id, size in community_sizes.items():
                comm_info = {
                    "id": comm_id,
                    "size": size
                }
                
                # Add node type information if available
                if comm_id in node_types:
                    comm_info["composition"] = node_types[comm_id]
                
                # Add key entities if available
                if str(comm_id) in key_entities or comm_id in key_entities:
                    comm_id_key = str(comm_id) if str(comm_id) in key_entities else comm_id
                    comm_info["key_entities"] = key_entities[comm_id_key]
                
                community_info.append(comm_info)
            
            # Create prompt for the LLM
            prompt = f"""
            You are analyzing the results of a graph community detection algorithm applied to a supply chain network.
            
            Graph Type: {graph_type}
            Number of Communities: {num_communities}
            Modularity Score: {modularity:.4f} (higher scores indicate stronger community structure)
            
            Community Details:
            {community_info}
            
            Based on this analysis, provide a business-oriented narrative explanation that:
            1. Explains what these communities represent in a {graph_type} context
            2. Highlights key insights and patterns
            3. Discusses business implications and potential actions
            4. Explains the significance of the modularity score
            
            Format the response with clear sections and bullet points where appropriate.
            Focus on practical insights that supply chain managers would find valuable.
            """
            
            # Get narrative from LLM
            response = processor.llm.explain_results(community_info, prompt)
            
            # Format the response for display
            formatted_response = f"<div class='narrative'>{response}</div>"
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"Error getting LLM narrative: {str(e)}")
            self.logger.error(traceback.format_exc())
            return f"<p>Error generating narrative: {str(e)}</p>"
        
    def generate_community_narrative(self, data: pd.DataFrame, 
                               communities: Dict[int, List], 
                               graph_type: str) -> str:
        """
        Generate business-relevant narrative for community detection results.
        This is a fallback method used when the LLM narrative generation fails.
        
        Args:
            data: DataFrame with community detection results
            communities: Dictionary mapping community IDs to lists of nodes
            graph_type: Type of graph for context-specific narrative
            
        Returns:
            HTML-formatted narrative text
        """
        try:
            num_communities = len(communities)
            community_sizes = {comm: len(nodes) for comm, nodes in communities.items()}
            
            # Basic stats
            largest_community = max(community_sizes.values()) if community_sizes else 0
            largest_comm_id = max(community_sizes, key=community_sizes.get) if community_sizes else 0
            small_communities = sum(1 for size in community_sizes.values() if size <= 3)
            
            # Get modularity score
            modularity = data.attrs.get('modularity', 0)
            
            # Context-specific narratives
            if "Supplier" in graph_type and "Part" in graph_type:
                narrative = f"""
                <div class="narrative">
                <h4>Supply Chain Cluster Analysis</h4>
                <ul>
                    <li>Found {num_communities} distinct clusters in your supplier-part network</li>
                    <li>The largest cluster (Community {largest_comm_id}) contains {largest_community} components ({(largest_community/len(data)*100):.1f}% of network)</li>
                    <li>{small_communities} small clusters indicate isolated supply relationships that may warrant closer review</li>
                    <li>Modularity score of {modularity:.4f} suggests {'strong' if modularity > 0.5 else 'moderate' if modularity > 0.3 else 'weak'} community structure</li>
                    <li>These clusters represent groups of suppliers and parts that are highly interconnected</li>
                </ul>
                <p><em>Note: Each community represents a group of suppliers and parts that are more densely connected to each other than to the rest of the network. This often indicates shared technological requirements, geographic proximity, or business relationships.</em></p>
                </div>
                """
            elif "Dependencies" in graph_type:
                narrative = f"""
                <div class="narrative">
                <h4>Part Dependency Cluster Analysis</h4>
                <ul>
                    <li>Identified {num_communities} distinct dependency clusters in your product architecture</li>
                    <li>The largest cluster (Community {largest_comm_id}) contains {largest_community} parts ({(largest_community/len(data)*100):.1f}% of all parts)</li>
                    <li>{small_communities} isolated component groups may indicate modular design opportunities</li>
                    <li>Modularity score of {modularity:.4f} suggests {'highly modular' if modularity > 0.5 else 'moderately modular' if modularity > 0.3 else 'tightly integrated'} product architecture</li>
                    <li>Each cluster represents components that are functionally related</li>
                </ul>
                <p><em>Note: Clusters in part dependencies often reveal functional modules within your product architecture. High modularity suggests opportunities for parallel development, while low modularity indicates tightly integrated components that may require coordinated changes.</em></p>
                </div>
                """
            elif "Risk" in graph_type:
                narrative = f"""
                <div class="narrative">
                <h4>Supplier Risk Cluster Analysis</h4>
                <ul>
                    <li>Detected {num_communities} risk clusters among your suppliers</li>
                    <li>The largest risk cluster (Community {largest_comm_id}) affects {largest_community} entities ({(largest_community/len(data)*100):.1f}% of network)</li>
                    <li>{small_communities} isolated risk clusters may indicate concentration of specific risk types</li>
                    <li>Modularity score of {modularity:.4f} suggests {'distinct' if modularity > 0.5 else 'somewhat overlapping' if modularity > 0.3 else 'highly interconnected'} risk exposures</li>
                    <li>Consider diversification strategies for large clusters to reduce correlated risks</li>
                </ul>
                <p><em>Note: Risk clusters identify groups of suppliers sharing similar risk profiles, often due to geographic proximity, similar business models, or shared dependencies. High modularity suggests well-defined risk zones that can be managed separately.</em></p>
                </div>
                """
            else:
                # Generic narrative for other graph types
                narrative = f"""
                <div class="narrative">
                <h4>Network Community Analysis</h4>
                <ul>
                    <li>Identified {num_communities} distinct communities in the network</li>
                    <li>The largest community contains {largest_community} nodes ({(largest_community/len(data)*100):.1f}% of network)</li>
                    <li>{small_communities} small communities (3 or fewer nodes) may represent specialized relationships</li>
                    <li>Modularity score of {modularity:.4f} indicates {'well-defined' if modularity > 0.5 else 'moderately defined' if modularity > 0.3 else 'weakly defined'} community structure</li>
                    <li>Communities represent groups of highly interconnected nodes</li>
                </ul>
                <p><em>Note: Network communities reveal natural groupings within your data based on connection patterns. These often correspond to real-world relationships or functional groupings that aren't immediately obvious in the raw data.</em></p>
                </div>
                """
            
            return narrative
        except Exception as e:
            self.logger.error(f"Error generating community narrative: {str(e)}")
            return f"<p>Error generating narrative: {str(e)}</p>"
    
    def get_centrality_narrative(self, data: pd.DataFrame, method: str, graph_type: str) -> str:
        """
        Generate narrative explanation for centrality analysis results.
        
        Args:
            data: DataFrame with centrality results
            method: Centrality method used
            graph_type: Type of graph analyzed
            
        Returns:
            HTML-formatted narrative text
        """
        try:
            from src.llm.query_processor import QueryProcessor
            
            # Prepare data for narrative
            if len(data) == 0:
                return "<p style='color:#333;'>No centrality data available for analysis.</p>"
            
                
            # Identify vertex column and centrality column
            vertex_col = 'vertex'
            centrality_cols = [col for col in data.columns if col != vertex_col]
            if not centrality_cols:
                return "<p style='color: #333333;'>No centrality metrics found in the results.</p>"
                
            centrality_col = centrality_cols[0]
            
            # Check data types
            if not pd.api.types.is_numeric_dtype(data[centrality_col]):
                self.logger.warning(f"Centrality column '{centrality_col}' is not numeric. Using fallback narrative.")
                return self.generate_centrality_narrative(data, method, graph_type)
            
            # Get top and bottom nodes
            try:
                top_5 = data.nlargest(min(5, len(data)), centrality_col)
                bottom_5 = data.nsmallest(min(5, len(data)), centrality_col)
            except Exception as e:
                self.logger.warning(f"Could not get top/bottom nodes: {str(e)}")
                return self.generate_centrality_narrative(data, method, graph_type)
            
            # Calculate basic stats
            total_nodes = len(data)
            avg_score = data[centrality_col].mean()
            max_score = data[centrality_col].max()
            min_score = data[centrality_col].min()
            
            # Format node lists for LLM prompt
            top_nodes = [{"id": str(row[vertex_col]), "score": float(row[centrality_col])} for _, row in top_5.iterrows()]
            bottom_nodes = [{"id": str(row[vertex_col]), "score": float(row[centrality_col])} for _, row in bottom_5.iterrows()]
            
            # Try to get narrative from LLM
            try:
                processor = QueryProcessor()
                
                # Prepare info for LLM
                centrality_info = {
                    "method": method,
                    "graph_type": graph_type,
                    "top_nodes": top_nodes,
                    "bottom_nodes": bottom_nodes,
                    "total_nodes": total_nodes,
                    "avg_score": float(avg_score),
                    "max_score": float(max_score),
                    "min_score": float(min_score)
                }
                
                # Generate prompt for LLM
                prompt = f"""
                You are analyzing the results of a {method} centrality analysis on a {graph_type} graph.
                
                Key Information:
                - Graph Type: {graph_type}
                - Centrality Method: {method}
                - Total Nodes: {total_nodes}
                - Average Score: {avg_score:.4f}
                - Maximum Score: {max_score:.4f}
                - Minimum Score: {min_score:.4f}
                
                Top 5 Nodes by Centrality:
                {top_nodes}
                
                Bottom 5 Nodes by Centrality:
                {bottom_nodes}
                
                Based on this centrality analysis, provide a business-oriented narrative explanation that:
                1. Explains what {method} centrality measures in the context of a {graph_type}
                2. Highlights the significance of the most central nodes
                3. Discusses potential vulnerabilities or bottlenecks in the supply chain
                4. Provides actionable insights for supply chain management
                
                Format the response with clear sections and bullet points where appropriate.
                Focus on practical insights that supply chain managers would find valuable.
                """
                
                # Get response from LLM as before...
                response = processor.llm.explain_results(centrality_info, prompt)
            
                # Add inline style to ensure text is visible
                response_with_styles = response.replace('<h4>', '<h4 style="color:#1f4e79;font-weight:bold;">')
                response_with_styles = response_with_styles.replace('<h5>', '<h5 style="color:#1f4e79;font-weight:bold;">')
                response_with_styles = response_with_styles.replace('<p>', '<p style="color:#333;">')
                response_with_styles = response_with_styles.replace('<li>', '<li style="color:#333;">')
                response_with_styles = response_with_styles.replace('<strong>', '<strong style="color:#1f4e79;font-weight:bold;">')
                
                # Return with a container div that has inline styles
                return f'<div style="color:#333;background-color:#f0f0f0;padding:15px;border-radius:5px;border-left:4px solid #4b778d;">{response_with_styles}</div>'
                
            except Exception as llm_error:
                self.logger.warning(f"LLM narrative generation failed: {str(llm_error)}")
                # Fall back to generated narrative
                return self.generate_centrality_narrative(data, method, graph_type)
                
        except Exception as e:
            self.logger.error(f"Error generating centrality narrative: {str(e)}")
            self.logger.error(traceback.format_exc())
            return f"<p style='color:#333;'>Error generating narrative: {str(e)}</p>"

       
    def generate_centrality_narrative(self, data: pd.DataFrame, method: str, graph_type: str) -> str:
        """
        Generate a fallback narrative for centrality analysis when LLM fails.
        
        Args:
            data: DataFrame with centrality results
            method: Centrality method used
            graph_type: Type of graph analyzed
            
        Returns:
            HTML-formatted narrative text
        """
        try:
            # Identify vertex column and centrality column
            vertex_col = 'vertex'
            centrality_cols = [col for col in data.columns if col != vertex_col]
            if not centrality_cols:
                return "<p style='color: #333333;'>No centrality metrics found in the results.</p>"
                
            centrality_col = centrality_cols[0]
            
            # Try to get top nodes safely
            top_nodes = []
            try:
                if pd.api.types.is_numeric_dtype(data[centrality_col]):
                    top_5 = data.nlargest(min(5, len(data)), centrality_col)
                    top_nodes = [str(row[vertex_col]) for _, row in top_5.iterrows()]
                else:
                    # If we can't sort, just take the first few rows
                    top_nodes = [str(row[vertex_col]) for _, row in data.head(5).iterrows()]
            except Exception as e:
                self.logger.warning(f"Could not extract top nodes: {str(e)}")
                # Just use first 5 rows if sorting fails
                top_nodes = [str(row[vertex_col]) for _, row in data.head(5).iterrows()]
                
            # Format method name for display
            display_method = method.replace('_', ' ').title()
            
            # Add consistent style attributes to all HTML elements
            style_attributes = "style='color: #333333;'"
            h4_style = "style='color: #1f4e79; font-weight: bold;'"
            h5_style = "style='color: #1f4e79; font-weight: bold;'"
            strong_style = "style='color: #1f4e79; font-weight: bold;'"
            
            # Prepare different narratives based on method and graph type
            if method.lower() in ['betweenness', 'betweenness_centrality']:
                if "Supplier" in graph_type and "Part" in graph_type:
                    narrative = f"""
                    <div {style_attributes}>
                    <h4 {h4_style}>Betweenness Centrality in Supplier-Part Network</h4>
                    <p {style_attributes}>Betweenness centrality identifies nodes that act as "bridges" between different parts of your supply chain network. Nodes with high betweenness often represent critical connectors that, if removed, could disconnect significant portions of your network.</p>
                    
                    <h5 {h5_style}>Key Findings</h5>
                    <ul {style_attributes}>
                        <li {style_attributes}>The analysis identified <strong {strong_style}>{top_nodes[0] if top_nodes else 'several nodes'}</strong> as critical connectors in your supply chain</li>
                        <li {style_attributes}>Other important connectors include {", ".join([f"<strong {strong_style}>{node}</strong>" for node in top_nodes[1:4]]) if len(top_nodes) > 1 else "various other components"}</li>
                        <li {style_attributes}>These components/suppliers are potential bottlenecks where delays or disruptions would have widespread impacts</li>
                    </ul>
                    
                    <h5 {h5_style}>Business Implications</h5>
                    <ul {style_attributes}>
                        <li {style_attributes}>Consider developing alternative sourcing or routing strategies for high-betweenness nodes</li>
                        <li {style_attributes}>Implement more stringent monitoring for these critical components or suppliers</li>
                        <li {style_attributes}>Evaluate opportunities to reduce dependency on these high-betweenness nodes through network redesign</li>
                    </ul>
                    </div>
                    """
                elif "Dependencies" in graph_type:
                    narrative = f"""
                    <div {style_attributes}>
                    <h4 {h4_style}>Betweenness Centrality in Part Dependencies</h4>
                    <p {style_attributes}>In the context of part dependencies, betweenness centrality highlights components that serve as critical links between different functional modules or subsystems in your product architecture.</p>
                    
                    <h5 {h5_style}>Key Findings</h5>
                    <ul {style_attributes}>
                        <li {style_attributes}>Component <strong {strong_style}>{top_nodes[0] if top_nodes else 'several components'}</strong> has high betweenness centrality, indicating it's a critical junction in your product architecture</li>
                        <li {style_attributes}>Other key connecting components include {", ".join([f"<strong {strong_style}>{node}</strong>" for node in top_nodes[1:4]]) if len(top_nodes) > 1 else "several other components"}</li>
                        <li {style_attributes}>These components likely connect different functional modules and could represent integration points</li>
                    </ul>
                    
                    <h5 {h5_style}>Business Implications</h5>
                    <ul {style_attributes}>
                        <li {style_attributes}>Focus quality assurance efforts on these critical components</li>
                        <li {style_attributes}>Consider standardizing interfaces around these components to improve modularity</li>
                        <li {style_attributes}>Evaluate whether redesign could reduce architectural complexity and risk</li>
                    </ul>
                    </div>
                    """
                else:
                    # Generic betweenness narrative
                    narrative = f"""
                    <div {style_attributes}>
                    <h4 {h4_style}>Betweenness Centrality Analysis</h4>
                    <p {style_attributes}>Betweenness centrality measures how often a node appears on the shortest paths between other nodes. High-betweenness nodes act as bridges or connectors in the network.</p>
                    
                    <h5 {h5_style}>Key Findings</h5>
                    <ul {style_attributes}>
                        <li {style_attributes}>Node <strong {strong_style}>{top_nodes[0] if top_nodes else 'several nodes'}</strong> has high betweenness centrality</li>
                        <li {style_attributes}>Other important connector nodes include {", ".join([f"<strong {strong_style}>{node}</strong>" for node in top_nodes[1:4]]) if len(top_nodes) > 1 else "various other nodes"}</li>
                        <li {style_attributes}>These nodes likely represent critical junctions in information or material flow</li>
                    </ul>
                    
                    <h5 {h5_style}>Business Implications</h5>
                    <ul {style_attributes}>
                        <li {style_attributes}>High betweenness nodes require special attention as their failure could disconnect parts of the network</li>
                        <li {style_attributes}>Consider redundancy strategies for these critical connection points</li>
                        <li {style_attributes}>Monitor these nodes for signs of stress or overload</li>
                    </ul>
                    </div>
                    """
            elif method.lower() in ['pagerank', 'pagerank_centrality']:
                # PageRank narrative
                narrative = f"""
                <div {style_attributes}>
                <h4 {h4_style}>PageRank Centrality Analysis</h4>
                <p {style_attributes}>PageRank measures the influence of nodes in the network based on their connections and the importance of nodes connecting to them. High PageRank nodes are connected to many other important nodes.</p>
                
                <h5 {h5_style}>Key Findings</h5>
                <ul {style_attributes}>
                    <li {style_attributes}>Node <strong {strong_style}>{top_nodes[0] if top_nodes else 'several nodes'}</strong> has high PageRank, indicating it's highly influential</li>
                    <li {style_attributes}>Other influential nodes include {", ".join([f"<strong {strong_style}>{node}</strong>" for node in top_nodes[1:4]]) if len(top_nodes) > 1 else "various other components"}</li>
                    <li {style_attributes}>These nodes likely represent components or suppliers with extensive connections throughout the network</li>
                </ul>
                
                <h5 {h5_style}>Business Implications</h5>
                <ul {style_attributes}>
                    <li {style_attributes}>High PageRank nodes can propagate effects (both positive and negative) throughout the network</li>
                    <li {style_attributes}>Quality improvements or disruptions at these nodes will have widespread impacts</li>
                    <li {style_attributes}>Consider these nodes as leverage points for implementing system-wide changes</li>
                </ul>
                </div>
                """
            elif method.lower() in ['eigenvector', 'eigenvector_centrality']:
                # Eigenvector narrative
                narrative = f"""
                <div {style_attributes}>
                <h4 {h4_style}>Eigenvector Centrality Analysis</h4>
                <p {style_attributes}>Eigenvector centrality measures node influence based on the principle that connections to high-scoring nodes contribute more to a node's score than connections to low-scoring nodes.</p>
                
                <h5 {h5_style}>Key Findings</h5>
                <ul {style_attributes}>
                    <li {style_attributes}>Node <strong {strong_style}>{top_nodes[0] if top_nodes else 'several nodes'}</strong> has high eigenvector centrality</li>
                    <li {style_attributes}>Other highly connected nodes include {", ".join([f"<strong {strong_style}>{node}</strong>" for node in top_nodes[1:4]]) if len(top_nodes) > 1 else "various other components"}</li>
                    <li {style_attributes}>These nodes are connected to many other well-connected nodes, making them strategically positioned</li>
                </ul>
                
                <h5 {h5_style}>Business Implications</h5>
                <ul {style_attributes}>
                    <li {style_attributes}>High eigenvector centrality nodes represent strategic leverage points in your network</li>
                    <li {style_attributes}>Focusing improvements on these nodes can have cascading positive effects</li>
                    <li {style_attributes}>Disruptions to these nodes might cause widespread secondary effects</li>
                    <li {style_attributes}>Consider building stronger relationships with suppliers identified as having high eigenvector centrality</li>
                </ul>
                </div>
                """
            else:
                # Generic centrality narrative
                narrative = f"""
                <div {style_attributes}>
                <h4 {h4_style}>{display_method} Analysis</h4>
                <p {style_attributes}>This centrality measure identifies the most important or influential nodes in your {graph_type} network.</p>
                
                <h5 {h5_style}>Key Findings</h5>
                <ul {style_attributes}>
                    <li {style_attributes}>Node <strong {strong_style}>{top_nodes[0] if top_nodes else 'several nodes'}</strong> has high centrality score</li>
                    <li {style_attributes}>Other important nodes include {", ".join([f"<strong {strong_style}>{node}</strong>" for node in top_nodes[1:4]]) if len(top_nodes) > 1 else "various other components"}</li>
                    <li {style_attributes}>These nodes likely play critical roles in the functioning of your supply chain or product architecture</li>
                </ul>
                
                <h5 {h5_style}>Business Implications</h5>
                <ul {style_attributes}>
                    <li {style_attributes}>Focus risk mitigation strategies on the most central nodes</li>
                    <li {style_attributes}>Consider redundancy or backup plans for critical components or suppliers</li>
                    <li {style_attributes}>Use this information to prioritize relationship management with key suppliers</li>
                </ul>
                </div>
                """
                
            return narrative
        except Exception as e:
            self.logger.error(f"Error generating centrality narrative: {str(e)}")
            return f"<p style='color: #333333;'>Error generating narrative: {str(e)}</p>"
    
    
    def visualize_centrality(self, data: pd.DataFrame, 
                       vertex_col: str = 'vertex', 
                       centrality_col: str = None,
                       top_n: int = 20) -> Optional[go.Figure]:
        """
        Create a visualization for centrality analysis results.
        
        Args:
            data: DataFrame with vertex and centrality columns
            vertex_col: Column name for vertex IDs
            centrality_col: Column name for centrality scores (auto-detected if None)
            top_n: Number of top vertices to show
            
        Returns:
            Plotly figure or None if visualization fails
        """
        try:
            if vertex_col not in data.columns:
                self.logger.error(f"Required column '{vertex_col}' not found")
                return None
            
            # Auto-detect centrality column if not specified
            if centrality_col is None:
                potential_cols = [col for col in data.columns if col != vertex_col]
                if potential_cols:
                    centrality_col = potential_cols[0]
                else:
                    self.logger.error("Could not auto-detect centrality column")
                    return None
            
            if centrality_col not in data.columns:
                self.logger.error(f"Centrality column '{centrality_col}' not found")
                return None
            
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Get actual supplier names from ArangoDB - ONLY ADDING THIS PART
            try:
                from src.utils.db_connection import ArangoDB
                db = ArangoDB()
                
                # Create a mapping of supplier IDs to names
                supplier_names = {}
                query = """
                FOR supplier IN suppliers
                    RETURN {
                        id: supplier.supplier_id,
                        name: supplier.supplier_name
                    }
                """
                cursor = db.db.aql.execute(query)
                for doc in cursor:
                    if 'id' in doc and 'name' in doc:
                        supplier_names[doc['id']] = doc['name']
                        
                self.logger.info(f"Loaded {len(supplier_names)} supplier names")
                
            except Exception as db_error:
                self.logger.warning(f"Could not load supplier names from database: {str(db_error)}")
                supplier_names = {}
            
            # Create better node labels - ONLY ADDING THIS PART
            node_labels = {}
            for idx, row in df.iterrows():
                node_id = str(row[vertex_col])
                # If it's a supplier ID that we have a name for
                if node_id.startswith('SUP') and node_id in supplier_names:
                    node_labels[node_id] = f"{supplier_names[node_id]} (Supplier)"
                # Otherwise use ID with type indicator based on prefix
                elif node_id.startswith('SUP'):
                    node_labels[node_id] = f"{node_id} (Supplier)"
                elif node_id.startswith('BAT'):
                    node_labels[node_id] = f"{node_id} (Battery)"
                elif node_id.startswith('MOT'):
                    node_labels[node_id] = f"{node_id} (Motor)"
                elif node_id.startswith('CAM'):
                    node_labels[node_id] = f"{node_id} (Camera)"
                elif node_id.startswith('GPU'):
                    node_labels[node_id] = f"{node_id} (GPU)"
                elif node_id.startswith('LID'):
                    node_labels[node_id] = f"{node_id} (LiDAR)"
                elif node_id.startswith('CTR'):
                    node_labels[node_id] = f"{node_id} (Controller)"
                elif node_id.startswith('INF'):
                    node_labels[node_id] = f"{node_id} (Infrared)"
                elif node_id.startswith('JNT'):
                    node_labels[node_id] = f"{node_id} (Joint)"
                elif node_id.startswith('CHS'):
                    node_labels[node_id] = f"{node_id} (Chassis)"
                else:
                    node_labels[node_id] = node_id
            
            # Apply labels to the dataframe
            df['node_label'] = df[vertex_col].apply(lambda x: node_labels.get(str(x), str(x)))
            
            # Continue with the original visualization code
            
            # Select top N vertices by centrality score
            try:
                # Sort by centrality value in descending order
                top_vertices = df.sort_values(by=centrality_col, ascending=False).head(top_n)
            except Exception as sort_error:
                self.logger.warning(f"Error sorting by centrality: {str(sort_error)}. Using simple sort.")
                # Try simple sort without key function
                try:
                    top_vertices = df.sort_values(by=centrality_col, ascending=False).head(top_n)
                except:
                    # Last resort: just take the first N rows
                    top_vertices = df.head(top_n)
            
            # Get centrality method name for title
            method_name = data.attrs.get('method', 'Centrality')
            method_name = method_name.replace('_', ' ').title()
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            # Add bars with better color handling
            fig.add_trace(go.Bar(
                y=top_vertices['node_label'],  # Use the node_label with supplier names 
                x=top_vertices[centrality_col],
                orientation='h',
                marker=dict(
                    color=top_vertices[centrality_col],
                    colorscale='Viridis',
                    colorbar=dict(
                        title=method_name + ' Score',
                        thickness=15,
                        x=1.02
                    )
                ),
                text=top_vertices[centrality_col].round(4),
                textposition='auto'
            ))
            
            # Enhance layout
            fig.update_layout(
                title=f'Top {len(top_vertices)} Nodes by {method_name}',
                xaxis_title=method_name + ' Score',
                yaxis_title='Node',
                height=max(self.default_height, 25 * len(top_vertices)),  # Adjust height based on number of nodes
                width=self.default_width,
                margin=dict(l=20, r=80, t=40, b=20),
                xaxis=dict(
                    showgrid=True,
                    zeroline=True,
                    zerolinewidth=1
                ),
                yaxis=dict(
                    automargin=True  # Give more space for labels
                )
            )
            
            # Handle empty results
            if len(top_vertices) == 0:
                fig.add_annotation(
                    text="No centrality data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
                
            return fig
            
        except Exception as e:
            self.logger.error(f"Error visualizing centrality: {str(e)}")
            self.logger.error(traceback.format_exc())
            return self.create_error_chart(f"Error visualizing centrality: {str(e)}")
    
    def visualize_shortest_paths(self, data: pd.DataFrame,
                               vertex_col: str = 'vertex',
                               distance_col: str = 'distance') -> Optional[go.Figure]:
        """
        Create a visualization for shortest path analysis results.
        
        Args:
            data: DataFrame with vertex and distance columns
            vertex_col: Column name for vertex IDs
            distance_col: Column name for distance values
            
        Returns:
            Plotly figure or None if visualization fails
        """
        try:
            if vertex_col not in data.columns or distance_col not in data.columns:
                self.logger.error(f"Required columns '{vertex_col}' or '{distance_col}' not found")
                return None
            
            # Sort by distance
            sorted_data = data.sort_values(distance_col)
            
            # Exclude infinity distances
            finite_data = sorted_data[sorted_data[distance_col] != float('inf')]
            
            # Get source vertex for title
            source_vertex = data.attrs.get('source_vertex', 'Unknown')
            
            # Create horizontal bar chart
            fig = px.bar(
                finite_data,
                y=vertex_col,
                x=distance_col,
                title=f'Shortest Path Distances from {source_vertex}',
                color=distance_col,
                color_continuous_scale=self.color_scale,
                height=self.default_height,
                width=self.default_width
            )
            
            # Add count of unreachable nodes if any
            unreachable_count = len(data) - len(finite_data)
            if unreachable_count > 0:
                fig.add_annotation(
                    text=f"{unreachable_count} nodes are unreachable",
                    xref="paper", yref="paper",
                    x=0.5, y=1.05,
                    showarrow=False
                )
            
            # Apply styling
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=20, r=20, t=80, b=20),
                xaxis_title='Distance',
                yaxis_title='Vertex ID',
                coloraxis_colorbar_title='Distance'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error visualizing shortest paths: {str(e)}")
            return None
    
    def create_network_graph(self, edges: pd.DataFrame, 
                           node_attrs: Optional[pd.DataFrame] = None,
                           source_col: str = 'source',
                           target_col: str = 'target',
                           weight_col: str = 'weight',
                           node_id_col: str = None,
                           layout_type: str = 'force') -> Optional[go.Figure]:
        """
        Create a network graph visualization from edge data.
        
        Args:
            edges: DataFrame with source, target, and optional weight columns
            node_attrs: Optional DataFrame with node attributes
            source_col: Column name for source vertices
            target_col: Column name for target vertices
            weight_col: Column name for edge weights
            node_id_col: Column name for node IDs in node_attrs
            layout_type: Layout algorithm ('force', 'circular', 'random')
            
        Returns:
            Plotly figure or None if visualization fails
        """
        try:
            import networkx as nx
            
            # Validate required columns
            if source_col not in edges.columns or target_col not in edges.columns:
                self.logger.error(f"Required columns '{source_col}' or '{target_col}' not found")
                return None
            
            # Create NetworkX graph
            G = nx.DiGraph() if 'directed' not in edges or edges['directed'].any() else nx.Graph()
            
            # Add edges
            for _, row in edges.iterrows():
                source = row[source_col]
                target = row[target_col]
                attrs = {}
                
                # Add weight if available
                if weight_col in edges.columns and not pd.isna(row[weight_col]):
                    attrs['weight'] = row[weight_col]
                
                # Add other attributes
                for col in row.index:
                    if col not in [source_col, target_col] and not pd.isna(row[col]):
                        attrs[col] = row[col]
                
                G.add_edge(source, target, **attrs)
            
            # Add node attributes if provided
            if node_attrs is not None and node_id_col is not None:
                for idx, row in node_attrs.iterrows():
                    node_id = row[node_id_col] if node_id_col in row else idx
                    if node_id in G.nodes:
                        for col in row.index:
                            if col != node_id_col and not pd.isna(row[col]):
                                G.nodes[node_id][col] = row[col]
            
            # Compute layout
            if layout_type == 'force':
                pos = nx.spring_layout(G)
            elif layout_type == 'circular':
                pos = nx.circular_layout(G)
            else:  # random
                pos = nx.random_layout(G)
            
            # Create edge trace
            edge_x = []
            edge_y = []
            edge_text = []
            
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # Create edge label
                attrs = {k: v for k, v in edge[2].items() if k != 'weight'}
                edge_text.append(f"{edge[0]} → {edge[1]}<br>" + "<br>".join(f"{k}: {v}" for k, v in attrs.items()))
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='text',
                text=edge_text,
                mode='lines'
            )
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Node label with attributes
                attrs = G.nodes[node]
                node_text.append(f"{node}<br>" + "<br>".join(f"{k}: {v}" for k, v in attrs.items()))
                
                # Node size based on degree
                node_size.append(5 + 3 * G.degree(node))
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale=self.color_scale,
                    size=node_size,
                    colorbar=dict(
                        thickness=15,
                        title='Node Connections',
                        xanchor='left',
                        titleside='right'
                    ),
                    line_width=2,
                    color=[G.degree(node) for node in G.nodes()]
                )
            )
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f'Network Graph ({len(G.nodes())} nodes, {len(G.edges())} edges)',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=self.default_height,
                    width=self.default_width
                )
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating network graph: {str(e)}")
            return None
    
    def create_error_chart(self, error_message: str) -> go.Figure:
        """
        Create an error message chart.
        
        Args:
            error_message: Error message to display
            
        Returns:
            Plotly figure with error message
        """
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        
        fig.update_layout(
            title="Error in Visualization",
            height=300,
            width=self.default_width,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig