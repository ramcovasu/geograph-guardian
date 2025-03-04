import cugraph
import cudf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback
from src.utils.logging import Logger
from src.utils.db_connection import ArangoDB

class GraphAnalytics:
    """GPU-accelerated graph analytics using NVIDIA cuGraph."""
    
    def __init__(self):
        """Initialize the GraphAnalytics class with logging."""
        self.logger = Logger().get_logger()
        self.db = ArangoDB()
        self.graph = None
        self.source_col = 'source'
        self.target_col = 'target'
        
        # Verify CUDA availability
        self._check_cuda_availability()
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available for graph processing."""
        try:
            # Create a small cuDF DataFrame to test CUDA availability
            test_df = cudf.DataFrame({'test': [1, 2, 3]})
            self.logger.info("CUDA is available for graph processing")
            return True
        except Exception as e:
            self.logger.warning(f"CUDA not available for graph processing: {str(e)}")
            return False
    
    def convert_to_cugraph(self, df: pd.DataFrame, source_col: str = None, target_col: str = None, 
                       weight_col: str = None, directed: bool = True) -> Optional[cugraph.Graph]:
        """
        Convert a pandas DataFrame to a cuGraph graph.
        
        Args:
            df: DataFrame containing graph edges
            source_col: Column name for source vertices
            target_col: Column name for target vertices
            weight_col: Column name for edge weights (optional)
            directed: Whether to create a directed graph
            
        Returns:
            cugraph.Graph or None if conversion fails
        """
        try:
            # Use provided column names or defaults
            source = source_col or self.source_col
            target = target_col or self.target_col
            
            # Validate required columns
            if source not in df.columns or target not in df.columns:
                self.logger.error(f"Required columns '{source}' and '{target}' not found in DataFrame")
                return None
            
            # Prepare DataFrame for cuGraph
            edge_data = df[[source, target]]
            if weight_col and weight_col in df.columns:
                edge_data[weight_col] = df[weight_col]
            
            # Convert to cuDF DataFrame
            cudf_df = cudf.DataFrame.from_pandas(edge_data)
            
            # Create cuGraph graph
            G = cugraph.Graph(directed=directed)
            
            # Add edges to graph
            if weight_col and weight_col in df.columns:
                G.from_cudf_edgelist(cudf_df, source=source, destination=target, edge_attr=weight_col)
            else:
                G.from_cudf_edgelist(cudf_df, source=source, destination=target)
            
            self.graph = G
            self.logger.info(f"Created cuGraph graph with {len(df)} edges")
            return G
            
        except Exception as e:
            self.logger.error(f"Error converting to cuGraph: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
        
    def run_community_detection(self, graph: Optional[cugraph.Graph] = None) -> Optional[pd.DataFrame]:
        """
        Run Louvain community detection algorithm to identify clusters.
        
        Args:
            graph: cuGraph graph (uses self.graph if None)
            
        Returns:
            DataFrame with vertex and partition columns or None if analysis fails
        """
        try:
            G = graph or self.graph
            if G is None:
                self.logger.error("No graph available for community detection")
                return None
            
            # Create an undirected graph if the input is directed
            # Louvain algorithm requires an undirected graph
            if G.is_directed():
                self.logger.info("Converting directed graph to undirected for community detection")
                
                # Get the edges DataFrame
                edges = G.edges()
                
                # Log the column names to debug
                self.logger.info(f"Edge DataFrame columns: {edges.columns.tolist()}")
                
                # Create a new undirected graph
                undirected_G = cugraph.Graph(directed=False)
                
                # For cuGraph 21.10+, column names are 'src' and 'dst'
                # For older versions, they might be 'source' and 'destination'
                # Let's check which columns exist and use them
                if 'src' in edges.columns and 'dst' in edges.columns:
                    source_col, dest_col = 'src', 'dst'
                elif 'source' in edges.columns and 'destination' in edges.columns:
                    source_col, dest_col = 'source', 'destination'
                else:
                    # If neither naming convention is found, use the first two columns
                    # This is a fallback and might not always work
                    cols = edges.columns.tolist()
                    if len(cols) >= 2:
                        source_col, dest_col = cols[0], cols[1]
                        self.logger.warning(f"Using columns {source_col} and {dest_col} as source and destination")
                    else:
                        raise ValueError("Could not identify source and destination columns in edge DataFrame")
                
                # If there's a weight column, include it
                if 'weight' in edges.columns:
                    undirected_G.from_cudf_edgelist(edges, source=source_col, destination=dest_col, edge_attr='weight')
                else:
                    undirected_G.from_cudf_edgelist(edges, source=source_col, destination=dest_col)
                    
                G = undirected_G
            
            # Run Louvain algorithm
            louvain_parts, modularity = cugraph.louvain(G)
            
            # Convert results to pandas DataFrame
            result_df = louvain_parts.to_pandas()
            
            # Add modularity score as metadata
            result_df.attrs['modularity'] = modularity
            
            self.logger.info(f"Louvain completed with modularity: {modularity}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Louvain community detection failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def run_centrality_analysis(self, graph: Optional[cugraph.Graph] = None, 
                           method: str = 'betweenness', k: int = 100) -> Optional[pd.DataFrame]:
        """
        Run centrality analysis to identify important nodes in the graph.
        
        Args:
            graph: cuGraph graph (uses self.graph if None)
            method: Centrality method ('betweenness', 'eigenvector', or 'pagerank')
            k: Number of vertices to sample for approximation
            
        Returns:
            DataFrame with vertex and centrality score or None if analysis fails
        """
        try:
            G = graph or self.graph
            if G is None:
                self.logger.error("No graph available for centrality analysis")
                return None
            
            # Get number of vertices in the graph to avoid sampling more than exist
            num_vertices = len(G.nodes())
            self.logger.info(f"Graph has {num_vertices} vertices")
            
            # Adjust k to not exceed number of vertices
            if k > num_vertices:
                self.logger.warning(f"Adjusting sample size from {k} to {num_vertices} to match graph size")
                k = num_vertices
            
            # Run appropriate centrality algorithm
            if method == 'betweenness':
                # Use approximation for large graphs, ensuring k doesn't exceed vertex count
                self.logger.info(f"Running betweenness centrality with k={k}")
                result = cugraph.betweenness_centrality(G, k=k, normalized=True)
                metric_name = 'betweenness_centrality'
            elif method == 'eigenvector':
                self.logger.info("Running eigenvector centrality")
                result = cugraph.eigenvector_centrality(G)
                metric_name = 'eigenvector_centrality'
            elif method == 'pagerank':
                self.logger.info("Running pagerank centrality")
                result = cugraph.pagerank(G)
                metric_name = 'pagerank'
            else:
                self.logger.error(f"Unsupported centrality method: {method}")
                return None
            
            # Convert results to pandas DataFrame
            result_df = result.to_pandas()
            result_df.attrs['method'] = method
            
            self.logger.info(f"{method.capitalize()} centrality computed successfully")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Centrality analysis failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # If betweenness centrality fails, try to switch to a different algorithm
            if method == 'betweenness':
                self.logger.info("Attempting to use pagerank as fallback centrality measure")
                try:
                    result = cugraph.pagerank(G)
                    result_df = result.to_pandas()
                    result_df.attrs['method'] = 'pagerank (fallback)'
                    self.logger.info("Successfully used pagerank as fallback")
                    return result_df
                except Exception as fallback_e:
                    self.logger.error(f"Fallback centrality also failed: {str(fallback_e)}")
            
            return None
    
    def run_shortest_path(self, source_vertex: Any, graph: Optional[cugraph.Graph] = None) -> Optional[pd.DataFrame]:
        """
        Run single-source shortest path analysis.
        
        Args:
            source_vertex: Source vertex for path calculation
            graph: cuGraph graph (uses self.graph if None)
            
        Returns:
            DataFrame with vertex and distance columns or None if analysis fails
        """
        try:
            G = graph or self.graph
            if G is None:
                self.logger.error("No graph available for shortest path analysis")
                return None
            
            # Check if source vertex exists in graph
            vertices = G.nodes().to_pandas()
            if source_vertex not in vertices.values:
                self.logger.error(f"Source vertex {source_vertex} not found in graph")
                return None
            
            # Run single-source shortest path algorithm
            distances = cugraph.sssp(G, source_vertex)
            
            # Convert results to pandas DataFrame
            result_df = distances.to_pandas()
            result_df.attrs['source_vertex'] = source_vertex
            
            self.logger.info(f"Shortest paths computed from {source_vertex}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Shortest path analysis failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def extract_graph_from_arangodb(self, query: str, source_field: str = 'source', 
                                   target_field: str = 'target', weight_field: str = None) -> Optional[pd.DataFrame]:
        """
        Extract graph data from ArangoDB using an AQL query.
        
        Args:
            query: AQL query to execute
            source_field: Field name for source vertex
            target_field: Field name for target vertex
            weight_field: Field name for edge weight (optional)
            
        Returns:
            DataFrame with graph data or None if extraction fails
        """
        try:
            cursor = self.db.db.aql.execute(query)
            data = [doc for doc in cursor]
            
            if not data:
                self.logger.warning("No data returned from ArangoDB query")
                return None
            
            df = pd.DataFrame(data)
            
            # Ensure required fields are present
            if source_field not in df.columns or target_field not in df.columns:
                self.logger.error(f"Required fields '{source_field}' and '{target_field}' not found in query results")
                return None
            
            # Rename columns to standard format if needed
            if source_field != self.source_col or target_field != self.target_col:
                df = df.rename(columns={
                    source_field: self.source_col,
                    target_field: self.target_col
                })
            
            # Rename weight field if present
            if weight_field and weight_field in df.columns and weight_field != 'weight':
                df = df.rename(columns={weight_field: 'weight'})
            
            self.logger.info(f"Extracted graph data with {len(df)} edges from ArangoDB")
            return df
            
        except Exception as e:
            self.logger.error(f"Error extracting graph from ArangoDB: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def get_graph_stats(self, graph: Optional[cugraph.Graph] = None) -> Dict[str, Any]:
        """
        Get basic statistics about the graph.
        
        Args:
            graph: cuGraph graph (uses self.graph if None)
            
        Returns:
            Dictionary with graph statistics
        """
        try:
            G = graph or self.graph
            if G is None:
                self.logger.error("No graph available for statistics")
                return {"error": "No graph available"}
            
            # Get basic graph statistics
            vertices = G.nodes().to_pandas()
            edges = G.edges().to_pandas()
            
            stats = {
                "num_vertices": len(vertices),
                "num_edges": len(edges),
                "is_directed": G.is_directed(),
                "density": 2 * len(edges) / (len(vertices) * (len(vertices) - 1)) if len(vertices) > 1 else 0
            }
            
            self.logger.info(f"Graph stats: {len(vertices)} vertices, {len(edges)} edges")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating graph statistics: {str(e)}")
            return {"error": str(e)}