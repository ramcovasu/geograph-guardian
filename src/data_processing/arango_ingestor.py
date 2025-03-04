import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from src.utils.db_connection import ArangoDB
from src.utils.logging import Logger

class ArangoIngestor:
    def __init__(self):
        self.logger = Logger().get_logger()
        self.db = ArangoDB()
        self.processed_path = Path(__file__).parent.parent.parent / 'data' / 'processed'
        self.graph_name = 'supplychain'

    def setup_graph(self):
        """Create and setup the graph with edge definitions."""
        try:
            # Delete existing graph if it exists
            if self.db.db.has_graph(self.graph_name):
                self.db.db.delete_graph(self.graph_name)
            
            # Create new graph
            graph = self.db.db.create_graph(self.graph_name)
            
            # Define edge definitions
            edge_definitions = [
                {
                    'edge_collection': 'supplier_provides_part',
                    'from_vertex_collections': ['suppliers'],
                    'to_vertex_collections': ['parts']
                },
                {
                    'edge_collection': 'part_depends_on',
                    'from_vertex_collections': ['parts'],
                    'to_vertex_collections': ['parts']
                },
                {
                    'edge_collection': 'supplier_orders',
                    'from_vertex_collections': ['suppliers'],
                    'to_vertex_collections': ['purchase_orders']
                },
                {
                    'edge_collection': 'part_transactions',
                    'from_vertex_collections': ['parts'],
                    'to_vertex_collections': ['inventory_transactions']
                }
            ]
            
            # Create collections and edge definitions
            for edge_def in edge_definitions:
                # Create edge collection
                if not self.db.db.has_collection(edge_def['edge_collection']):
                    self.db.db.create_collection(edge_def['edge_collection'], edge=True)
                
                # Create vertex collections
                for collection in edge_def['from_vertex_collections'] + edge_def['to_vertex_collections']:
                    if not self.db.db.has_collection(collection):
                        self.db.db.create_collection(collection)
                
                # Add edge definition to graph
                graph.create_edge_definition(
                    edge_collection=edge_def['edge_collection'],
                    from_vertex_collections=edge_def['from_vertex_collections'],
                    to_vertex_collections=edge_def['to_vertex_collections']
                )
                
            self.logger.info(f"Successfully set up graph '{self.graph_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up graph: {str(e)}")
            raise
            
    def cleanup_database(self):
        """Clean up existing collections before new ingestion."""
        try:
            # Drop the graph first if it exists
            if self.db.db.has_graph(self.graph_name):
                self.db.db.delete_graph(self.graph_name)
            
            # Drop collections
            collections = self.db.db.collections()
            for collection in collections:
                if not collection['name'].startswith('_'):
                    self.logger.info(f"Dropping collection: {collection['name']}")
                    self.db.db.delete_collection(collection['name'])
                    
            self.logger.info("Database cleanup completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error during database cleanup: {str(e)}")
            return False

    def create_collection(self, name: str, edge: bool = False):
        """Create a collection if it doesn't exist, or recreate if wrong type."""
        try:
            if self.db.db.has_collection(name):
                collection = self.db.db.collection(name)
                is_edge_collection = collection.properties().get('type') == 3
                
                # If collection exists but is wrong type, recreate it
                if edge != is_edge_collection:
                    self.logger.warning(f"Collection {name} exists with wrong type. Recreating...")
                    self.db.db.delete_collection(name)
                    collection = self.db.db.create_collection(name, edge=edge)
            else:
                collection = self.db.db.create_collection(name, edge=edge)
                
            return collection
        except Exception as e:
            raise Exception(f"Error creating collection {name}: {str(e)}")

    def batch_insert_documents(self, collection, docs: List[Dict], batch_size: int = 1000):
        """Insert documents in batches."""
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            try:
                collection.import_bulk(batch, on_duplicate="update")
                self.logger.info(f"Inserted batch of {len(batch)} documents")
            except Exception as e:
                self.logger.error(f"Error inserting batch: {str(e)}")
                raise

    def ingest_file(self, filename: str, ingestion_config: Dict[str, Any]) -> bool:
        """Ingest a processed file into ArangoDB."""
        try:
            # Read processed data
            file_path = self.processed_path / filename
            df = pd.read_csv(file_path)
            self.logger.info(f"Read processed {filename} with {len(df)} rows")

            # Get or create collection
            collection_name = ingestion_config['collection_name']
            is_edge = ingestion_config.get('is_edge', False)
            collection = self.create_collection(collection_name, edge=is_edge)

            # Prepare documents
            docs = []
            for _, row in df.iterrows():
                doc = row.to_dict()
                
                # Remove any NaN values
                doc = {k: v for k, v in doc.items() if pd.notna(v)}

                if is_edge:
                    # Validate required fields for edges
                    if pd.notna(doc.get(ingestion_config['from_field'])) and pd.notna(doc.get(ingestion_config['to_field'])):
                        # Create composite key for edges
                        doc['_key'] = f"{doc[ingestion_config['from_field']]}_{doc[ingestion_config['to_field']]}"
                        
                        # Set _from and _to fields
                        doc['_from'] = f"{ingestion_config['from_collection']}/{str(doc[ingestion_config['from_field']])}"
                        doc['_to'] = f"{ingestion_config['to_collection']}/{str(doc[ingestion_config['to_field']])}"
                        docs.append(doc)
                    else:
                        self.logger.warning(f"Skipping edge due to missing fields: {doc}")
                else:
                    # Handle document collections
                    key_field = ingestion_config['key_field']
                    if pd.notna(doc.get(key_field)):
                        doc['_key'] = str(doc[key_field])
                        docs.append(doc)
                    else:
                        self.logger.warning(f"Skipping document due to missing key: {doc}")

            # Batch insert documents
            if docs:
                self.batch_insert_documents(collection, docs)
                
                # Verify edge creation if it's an edge collection
                if is_edge:
                    sample = self.verify_edge_collection(collection_name)
                    if not sample:
                        raise Exception(f"Edge verification failed for {collection_name}")
                        
                return True
            else:
                self.logger.warning(f"No valid documents to insert for {filename}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error ingesting {filename}: {str(e)}")
            return False

    def verify_edge_collection(self, collection_name: str) -> bool:
        """Verify that edges were properly created in the collection."""
        try:
            query = f"""
            FOR doc IN {collection_name}
            LIMIT 1
            RETURN {{
                _from: doc._from,
                _to: doc._to,
                _key: doc._key
            }}
            """
            cursor = self.db.db.aql.execute(query)
            result = list(cursor)
            
            if result and '_from' in result[0] and '_to' in result[0]:
                self.logger.info(f"Edge verification successful for {collection_name}")
                self.logger.info(f"Sample edge: {result[0]}")
                return True
            else:
                self.logger.error(f"Edge verification failed for {collection_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error verifying edges in {collection_name}: {str(e)}")
            return False

def get_ingestion_configs() -> Dict[str, Dict[str, Any]]:
    """Define ingestion configurations for each file."""
    return {
        # Vertex Collections
        'products.csv': {
            'collection_name': 'products',
            'key_field': 'product_id',
            'indexes': [
                {'fields': ['product_category'], 'type': 'persistent'}
            ]
        },
        
        'product_parts.csv': {
            'collection_name': 'parts',
            'key_field': 'part_id',
            'indexes': [
                {'fields': ['part_category'], 'type': 'persistent'},
                {'fields': ['criticality_level'], 'type': 'persistent'}
            ]
        },
        
        'supplier_master.csv': {
            'collection_name': 'suppliers',
            'key_field': 'supplier_id',
            'indexes': [
                {'fields': ['country'], 'type': 'persistent'},
                {'fields': ['supplier_rating'], 'type': 'persistent'}
            ]
        },

        # Edge Collections
        'supplier_parts.csv': {
            'collection_name': 'supplier_provides_part',
            'is_edge': True,
            'key_field': 'supplier_id',
            'from_collection': 'suppliers',
            'from_field': 'supplier_id',
            'to_collection': 'parts',
            'to_field': 'part_id',
            'indexes': [
                {'fields': ['is_primary_supplier'], 'type': 'persistent'}
            ]
        },
        
        'part_dependencies.csv': {
            'collection_name': 'part_depends_on',
            'is_edge': True,
            'key_field': 'part_id',
            'from_collection': 'parts',
            'from_field': 'part_id',
            'to_collection': 'parts',
            'to_field': 'dependent_part_id',
            'indexes': [
                {'fields': ['dependency_type'], 'type': 'persistent'}
            ]
        },

        # Regular Collections
        'inventory_current.csv': {
            'collection_name': 'inventory',
            'key_field': 'part_id',
            'indexes': [
                {'fields': ['warehouse_location'], 'type': 'persistent'},
                {'fields': ['quantity_on_hand'], 'type': 'persistent'}
            ]
        },
        
        'purchase_orders.csv': {
            'collection_name': 'purchase_orders',
            'key_field': 'po_id',
            'indexes': [
                {'fields': ['supplier_id'], 'type': 'persistent'},
                {'fields': ['part_id'], 'type': 'persistent'},
                {'fields': ['status'], 'type': 'persistent'},
                {'fields': ['order_date'], 'type': 'persistent'}
            ]
        },
        
        'supplier_risk_factors.csv': {
            'collection_name': 'risk_factors',
            'key_field': 'supplier_id',
            'indexes': [
                {'fields': ['risk_category'], 'type': 'persistent'},
                {'fields': ['risk_score'], 'type': 'persistent'}
            ]
        },
        
        'inventory_transactions.csv': {
            'collection_name': 'inventory_transactions',
            'key_field': 'transaction_id',
            'indexes': [
                {'fields': ['part_id'], 'type': 'persistent'},
                {'fields': ['transaction_date'], 'type': 'persistent'},
                {'fields': ['transaction_type'], 'type': 'persistent'}
            ]
        }
    }

def ingest_all_files():
    """Ingest all processed files into ArangoDB."""
    ingestor = ArangoIngestor()
    
    try:
        # Clean up existing data
        if not ingestor.cleanup_database():
            raise Exception("Database cleanup failed")
        
        # Setup graph first!
        ingestor.logger.info("Setting up graph structure...")
        ingestor.setup_graph()
        
        configs = get_ingestion_configs()
        
        # Process vertex collections first, then edge collections
        vertex_files = [f for f, c in configs.items() if not c.get('is_edge', False)]
        edge_files = [f for f, c in configs.items() if c.get('is_edge', False)]
        
        results = {}
        
        # Ingest vertices first
        ingestor.logger.info("Ingesting vertex collections...")
        for filename in vertex_files:
            config = configs[filename]
            results[filename] = ingestor.ingest_file(filename, config)
        
        # Then ingest edges
        ingestor.logger.info("Ingesting edge collections...")
        for filename in edge_files:
            config = configs[filename]
            results[filename] = ingestor.ingest_file(filename, config)
        
        ingestor.logger.info("All files ingested successfully")
        return results
        
    except Exception as e:
        ingestor.logger.error(f"Error in ingestion process: {str(e)}")
        raise

if __name__ == "__main__":
    results = ingest_all_files()
    for filename, success in results.items():
        print(f"{filename}: {'Success' if success else 'Failed'}")

# These exports are now explicitly available at the module level
__all__ = ['ArangoIngestor', 'get_ingestion_configs', 'ingest_all_files']