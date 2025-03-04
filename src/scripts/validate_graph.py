from src.utils.db_connection import ArangoDB
from src.utils.logging import Logger
from typing import List, Dict, Any

class GraphValidator:
    def __init__(self):
        self.logger = Logger().get_logger()
        self.db = ArangoDB()
        self.graph_name = 'supplychain'

    def _get_sample_edges(self, collection_name: str) -> List[Dict[str, Any]]:
        """Get sample edges from a collection using AQL."""
        query = f"FOR doc IN {collection_name} LIMIT 1 RETURN doc"
        cursor = self.db.db.aql.execute(query)
        return list(cursor)

    def validate_graph_setup(self) -> bool:
        """Validate the graph setup and relationships."""
        self.logger.info("=== Validating Graph Setup ===")
        
        try:
            # Check if graph exists
            if not self.db.db.has_graph(self.graph_name):
                self.logger.error(f"Graph '{self.graph_name}' not found!")
                return False
                
            graph = self.db.db.graph(self.graph_name)
            
            # Validate edge definitions
            edge_definitions = graph.edge_definitions()
            self.logger.info("\nEdge Definitions:")
            for edge_def in edge_definitions:
                self.logger.info(f"\nEdge Collection: {edge_def['edge_collection']}")
                self.logger.info(f"From Collections: {edge_def['from_vertex_collections']}")
                self.logger.info(f"To Collections: {edge_def['to_vertex_collections']}")

            # Validate collections exist
            self.logger.info("\nVerifying Collections:")
            collections = self.db.db.collections()
            for coll in collections:
                if not coll['name'].startswith('_'):
                    self.logger.info(f"Collection: {coll['name']}, Type: {'edge' if coll['type'] == 3 else 'document'}")

            # Validate edge data
            self.logger.info("\nValidating Edge Data:")
            
            # Check supplier_provides_part edges
            supplier_edges = self._get_sample_edges('supplier_provides_part')
            if supplier_edges:
                edge = supplier_edges[0]
                self.logger.info("\nSupplier-Part Edge Sample:")
                self.logger.info(f"From: {edge['_from']}")
                self.logger.info(f"To: {edge['_to']}")
                self.logger.info(f"Properties: {[k for k in edge.keys() if not k.startswith('_')]}")
            else:
                self.logger.warning("No supplier-part relationships found")

            # Check part_depends_on edges
            part_edges = self._get_sample_edges('part_depends_on')
            if part_edges:
                edge = part_edges[0]
                self.logger.info("\nPart Dependency Edge Sample:")
                self.logger.info(f"From: {edge['_from']}")
                self.logger.info(f"To: {edge['_to']}")
                self.logger.info(f"Properties: {[k for k in edge.keys() if not k.startswith('_')]}")
            else:
                self.logger.warning("No part dependency relationships found")

            # Validate vertex connections
            self.logger.info("\nValidating Vertex Connections:")
            
            # Check supplier connections
            supplier_query = """
            FOR s IN suppliers
                LET parts = (
                    FOR e IN supplier_provides_part
                    FILTER e._from == s._id
                    RETURN e._to
                )
                LIMIT 1
                RETURN {
                    supplier: s._id,
                    connected_parts: parts
                }
            """
            supplier_connections = list(self.db.db.aql.execute(supplier_query))
            if supplier_connections:
                self.logger.info(f"Supplier Connection Sample: {supplier_connections[0]}")
            else:
                self.logger.warning("No supplier connections found")

            self.logger.info("\n=== Graph Validation Complete ===")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return False

def main():
    validator = GraphValidator()
    success = validator.validate_graph_setup()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()