from src.utils.db_connection import ArangoDB
from src.utils.logging import Logger

class EdgeValidator:
    def __init__(self):
        self.db = ArangoDB()
        self.logger = Logger().get_logger()

    def validate_edge_collections(self):
        """Validate edge collections and their connections."""
        edge_collections = {
            'supplier_provides_part': {
                'from_collection': 'suppliers',
                'to_collection': 'parts'
            },
            'part_depends_on': {
                'from_collection': 'parts',
                'to_collection': 'parts'
            }
        }

        results = {}
        
        for edge_col, config in edge_collections.items():
            self.logger.info(f"\nValidating edge collection: {edge_col}")
            
            # Check if collection exists and is edge type
            if not self.db.db.has_collection(edge_col):
                self.logger.error(f"Collection {edge_col} does not exist!")
                continue
                
            collection = self.db.db.collection(edge_col)
            if collection.properties()['type'] != 3:  # type 3 is for edge collections
                self.logger.error(f"Collection {edge_col} is not an edge collection!")
                continue

            # Check edge count
            edge_count = collection.count()
            self.logger.info(f"Total edges: {edge_count}")

            # Sample edge validation
            aql = f"""
            FOR e IN {edge_col}
            LIMIT 1
            RETURN {{
                edge: e,
                from_exists: LENGTH(
                    FOR v IN {config['from_collection']}
                    FILTER v._id == e._from
                    RETURN v
                ),
                to_exists: LENGTH(
                    FOR v IN {config['to_collection']}
                    FILTER v._id == e._to
                    RETURN v
                )
            }}
            """
            cursor = self.db.db.aql.execute(aql)
            sample = next(cursor, None)
            
            if sample:
                self.logger.info("Sample edge structure:")
                self.logger.info(f"  _from: {sample['edge'].get('_from')}")
                self.logger.info(f"  _to: {sample['edge'].get('_to')}")
                self.logger.info(f"From vertex exists: {bool(sample['from_exists'])}")
                self.logger.info(f"To vertex exists: {bool(sample['to_exists'])}")
            else:
                self.logger.error("No edges found in collection!")
                continue

            # Validate all edges have proper connections
            aql = f"""
            LET invalid_edges = (
                FOR e IN {edge_col}
                LET from_exists = LENGTH(
                    FOR v IN {config['from_collection']}
                    FILTER v._id == e._from
                    RETURN v
                )
                LET to_exists = LENGTH(
                    FOR v IN {config['to_collection']}
                    FILTER v._id == e._to
                    RETURN v
                )
                FILTER from_exists == 0 OR to_exists == 0
                RETURN e._key
            )
            RETURN {{
                "total_edges": LENGTH(@@collection),
                "invalid_edges": invalid_edges,
                "invalid_count": LENGTH(invalid_edges)
            }}
            """
            
            # Execute with bind vars to safely reference collection
            cursor = self.db.db.aql.execute(
                aql,
                bind_vars={'@collection': edge_col}
            )
            stats = next(cursor)
            
            results[edge_col] = {
                'total_edges': stats['total_edges'],
                'invalid_edges': stats['invalid_count'],
                'is_valid': stats['invalid_count'] == 0
            }

            self.logger.info(f"\nValidation Results for {edge_col}:")
            self.logger.info(f"Total Edges: {stats['total_edges']}")
            self.logger.info(f"Invalid Edges: {stats['invalid_count']}")
            
            if stats['invalid_count'] > 0:
                self.logger.error(f"Found {stats['invalid_count']} invalid edges!")
                self.logger.error(f"Invalid edge keys: {stats['invalid_edges']}")
            else:
                self.logger.info("All edges are valid!")

        return results

    def print_summary(self, results):
        """Print validation summary."""
        self.logger.info("\n=== Edge Validation Summary ===")
        all_valid = True
        
        for collection, result in results.items():
            status = "✓" if result['is_valid'] else "✗"
            self.logger.info(f"{collection}: {status}")
            self.logger.info(f"  Total Edges: {result['total_edges']}")
            self.logger.info(f"  Invalid Edges: {result['invalid_edges']}")
            
            if not result['is_valid']:
                all_valid = False
                
        return all_valid

if __name__ == "__main__":
    validator = EdgeValidator()
    results = validator.validate_edge_collections()
    all_valid = validator.print_summary(results)
    
    # Exit with appropriate status code
    exit(0 if all_valid else 1)