import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback
from src.utils.logging import Logger
from src.utils.db_connection import ArangoDB
from src.llm.query_processor import QueryProcessor

class GraphDataConverter:
    """Converts data between ArangoDB and formats suitable for graph analytics."""
    
    def __init__(self):
        """Initialize the GraphDataConverter class."""
        self.logger = Logger().get_logger()
        self.db = ArangoDB()
        self.query_processor = QueryProcessor()
    
    def prepare_supplier_part_graph(self) -> Optional[pd.DataFrame]:
        """
        Prepare a graph of supplier-part relationships from ArangoDB.
        
        Returns:
            DataFrame with supplier-part edges or None if preparation fails
        """
        try:
            # Query to extract supplier-part relationships with names
            query = """
            FOR sp IN supplier_provides_part
                LET supplier = DOCUMENT(sp._from)
                LET part = DOCUMENT(sp._to)
                RETURN {
                    source: supplier.supplier_id,
                    source_name: supplier.supplier_name,
                    source_type: "supplier",
                    source_country: supplier.country,
                    source_rating: supplier.supplier_rating,
                    target: part.part_id,
                    target_name: part.part_name,
                    target_type: "part",
                    target_category: part.part_category,
                    target_criticality: part.criticality_level,
                    is_primary: sp.is_primary_supplier,
                    lead_time: sp.lead_time_days,
                    cost: sp.unit_cost
                }
            """
            
            cursor = self.db.db.aql.execute(query)
            data = [doc for doc in cursor]
            
            if not data:
                self.logger.warning("No supplier-part relationships found")
                return None
            
            df = pd.DataFrame(data)
            
            # Convert boolean to numeric weight (primary suppliers have higher weight)
            if 'is_primary' in df.columns:
                df['weight'] = df['is_primary'].apply(lambda x: 2.0 if x else 1.0)
            
            # Create node metadata dictionary for visualization
            source_meta = {row['source']: {
                'name': row['source_name'],
                'type': row['source_type'],
                'country': row.get('source_country', ''),
                'rating': row.get('source_rating', '')
            } for _, row in df.iterrows()}
            
            target_meta = {row['target']: {
                'name': row['target_name'],
                'type': row['target_type'],
                'category': row.get('target_category', ''),
                'criticality': row.get('target_criticality', '')
            } for _, row in df.iterrows()}
            
            # Combine metadata
            node_metadata = {**source_meta, **target_meta}
            
            # Store metadata in DataFrame attributes for use in visualization
            df.attrs['node_metadata'] = node_metadata
            
            self.logger.info(f"Prepared supplier-part graph with {len(df)} edges")
            return df

            
        except Exception as e:
            self.logger.error(f"Error preparing supplier-part graph: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def prepare_part_dependency_graph(self) -> Optional[pd.DataFrame]:
        """
        Prepare a graph of part dependencies from ArangoDB.
        
        Returns:
            DataFrame with part dependency edges or None if preparation fails
        """
        try:
            # Query to extract part dependencies with names
            query = """
            FOR pd IN part_depends_on
                LET from_part = DOCUMENT(pd._from)
                LET to_part = DOCUMENT(pd._to)
                RETURN {
                    source: from_part.part_id,
                    source_name: from_part.part_name,
                    source_category: from_part.part_category,
                    source_criticality: from_part.criticality_level,
                    target: to_part.part_id,
                    target_name: to_part.part_name,
                    target_category: to_part.part_category,
                    target_criticality: to_part.criticality_level,
                    dependency_type: pd.dependency_type,
                    description: pd.description
                }
            """
            
            cursor = self.db.db.aql.execute(query)
            data = [doc for doc in cursor]
            
            if not data:
                self.logger.warning("No part dependencies found")
                return None
            
            df = pd.DataFrame(data)
            
            # Map dependency types to numeric weights
            if 'dependency_type' in df.columns:
                dependency_weights = {
                    'REQUIRED': 3.0,
                    'OPTIONAL': 1.0,
                    'ALTERNATIVE': 0.5,
                    'UNKNOWN': 1.0
                }
                df['weight'] = df['dependency_type'].map(lambda x: dependency_weights.get(x, 1.0))
            
            # Create node metadata dictionary for visualization
            source_meta = {row['source']: {
                'name': row['source_name'],
                'type': 'part',
                'category': row.get('source_category', ''),
                'criticality': row.get('source_criticality', '')
            } for _, row in df.iterrows()}
            
            target_meta = {row['target']: {
                'name': row['target_name'],
                'type': 'part',
                'category': row.get('target_category', ''),
                'criticality': row.get('target_criticality', '')
            } for _, row in df.iterrows()}
            
            # Combine metadata
            node_metadata = {**source_meta, **target_meta}
            
            # Store metadata in DataFrame attributes for use in visualization
            df.attrs['node_metadata'] = node_metadata
            
            self.logger.info(f"Prepared part dependency graph with {len(df)} edges")
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing supplier risk graph: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def prepare_combined_supply_chain_graph(self) -> Optional[pd.DataFrame]:
        """
        Prepare a comprehensive supply chain graph combining multiple relationships.
        
        Returns:
            DataFrame with combined graph edges or None if preparation fails
        """
        try:
            # Get individual graphs
            supplier_part_df = self.prepare_supplier_part_graph()
            part_dependency_df = self.prepare_part_dependency_graph()
            
            if supplier_part_df is None and part_dependency_df is None:
                self.logger.error("No graph data available")
                return None
            
            # Combine graphs
            dfs_to_combine = []
            
            if supplier_part_df is not None:
                # Ensure standard columns
                supplier_part_df = supplier_part_df[['source', 'target', 'weight']]
                dfs_to_combine.append(supplier_part_df)
            
            if part_dependency_df is not None:
                # Ensure standard columns
                part_dependency_df = part_dependency_df[['source', 'target', 'weight']]
                dfs_to_combine.append(part_dependency_df)
            
            # Combine dataframes
            combined_df = pd.concat(dfs_to_combine, ignore_index=True)
            
            self.logger.info(f"Prepared combined supply chain graph with {len(combined_df)} edges")
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error preparing combined supply chain graph: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def extract_graph_from_query(self, query: str) -> Optional[pd.DataFrame]:
        """
        Extract graph data from a custom AQL query.
        
        Args:
            query: AQL query string
            
        Returns:
            DataFrame with graph data or None if extraction fails
        """
        try:
            # Execute the query
            cursor = self.db.db.aql.execute(query)
            data = [doc for doc in cursor]
            
            if not data:
                self.logger.warning("No data returned from query")
                return None
            
            df = pd.DataFrame(data)
            
            # Check if minimal graph data exists
            required_cols = ['source', 'target']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Query results missing required columns: {required_cols}")
                return None
            
            self.logger.info(f"Extracted graph from custom query with {len(df)} edges")
            return df
            
        except Exception as e:
            self.logger.error(f"Error extracting graph from query: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def convert_arango_edges_to_df(self, edge_collection: str, 
                                 from_prefix: str = None, 
                                 to_prefix: str = None) -> Optional[pd.DataFrame]:
        """
        Convert an ArangoDB edge collection to a DataFrame suitable for graph analysis.
        
        Args:
            edge_collection: Name of the edge collection
            from_prefix: Prefix to strip from _from values (e.g., 'suppliers/')
            to_prefix: Prefix to strip from _to values (e.g., 'parts/')
            
        Returns:
            DataFrame with source and target columns or None if conversion fails
        """
        try:
            query = f"""
            FOR edge IN {edge_collection}
                RETURN {{
                    source: edge._from,
                    target: edge._to
                }}
            """
            
            cursor = self.db.db.aql.execute(query)
            data = [doc for doc in cursor]
            
            if not data:
                self.logger.warning(f"No edges found in collection {edge_collection}")
                return None
            
            df = pd.DataFrame(data)
            
            # Process _from and _to fields to extract IDs
            if from_prefix:
                df['source'] = df['source'].str.replace(from_prefix, '', regex=False)
            
            if to_prefix:
                df['target'] = df['target'].str.replace(to_prefix, '', regex=False)
            
            self.logger.info(f"Converted {edge_collection} to DataFrame with {len(df)} edges")
            return df
            
        except Exception as e:
            self.logger.error(f"Error converting edge collection to DataFrame: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def extract_node_attributes(self, collection: str, id_field: str) -> Optional[pd.DataFrame]:
        """
        Extract node attributes from a collection for enriching graph data.
        
        Args:
            collection: Name of the collection
            id_field: Field to use as node ID
            
        Returns:
            DataFrame with node attributes or None if extraction fails
        """
        try:
            query = f"""
            FOR doc IN {collection}
                RETURN doc
            """
            
            cursor = self.db.db.aql.execute(query)
            data = [doc for doc in cursor]
            
            if not data:
                self.logger.warning(f"No documents found in collection {collection}")
                return None
            
            df = pd.DataFrame(data)
            
            # Ensure id_field is present
            if id_field not in df.columns:
                self.logger.error(f"ID field '{id_field}' not found in collection {collection}")
                return None
            
            # Set id_field as index for easier joining later
            df = df.set_index(id_field)
            
            self.logger.info(f"Extracted attributes for {len(df)} nodes from {collection}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error extracting node attributes: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None