from src.utils.db_connection import ArangoDB
from src.utils.logging import Logger

class DataValidator:
    def __init__(self):
        self.db = ArangoDB()
        self.logger = Logger().get_logger()

    def validate_all_collections(self):
        """Validate counts and basic data in all collections."""
        collections = {
            'Vertices': [
                'products',
                'parts',
                'suppliers',
                'purchase_orders',
                'risk_factors',
                'inventory',
                'inventory_transactions'
            ],
            'Edges': [
                'supplier_provides_part',
                'part_depends_on'
            ]
        }

        results = {}
        
        self.logger.info("\n=== Collection Counts ===")
        for collection_type, collection_list in collections.items():
            self.logger.info(f"\n{collection_type}:")
            for collection_name in collection_list:
                try:
                    collection = self.db.db.collection(collection_name)
                    count = collection.count()
                    results[collection_name] = count
                    self.logger.info(f"{collection_name}: {count} documents")
                except Exception as e:
                    self.logger.error(f"Error checking {collection_name}: {str(e)}")
                    results[collection_name] = -1

        return results

    def validate_relationships(self):
        """Validate relationships between collections."""
        self.logger.info("\n=== Relationship Validation ===")
        
        # Check supplier-part relationships
        aql = """
        LET supplier_parts = (
            FOR sp IN supplier_provides_part
            RETURN {
                supplier: sp._from,
                part: sp._to
            }
        )
        RETURN {
            total_relationships: LENGTH(supplier_parts),
            unique_suppliers: LENGTH(UNIQUE(supplier_parts[* RETURN PARSE_IDENTIFIER(CURRENT.supplier).key])),
            unique_parts: LENGTH(UNIQUE(supplier_parts[* RETURN PARSE_IDENTIFIER(CURRENT.part).key]))
        }
        """
        try:
            cursor = self.db.db.aql.execute(aql)
            stats = next(cursor)
            self.logger.info("Supplier-Part Relationships:")
            self.logger.info(f"Total relationships: {stats['total_relationships']}")
            self.logger.info(f"Unique suppliers: {stats['unique_suppliers']}")
            self.logger.info(f"Unique parts: {stats['unique_parts']}")
        except Exception as e:
            self.logger.error(f"Error validating supplier-part relationships: {str(e)}")

    def validate_purchase_orders(self):
        """Validate purchase order data."""
        self.logger.info("\n=== Purchase Orders Validation ===")
        
        aql = """
        RETURN {
            total_pos: LENGTH(purchase_orders),
            status_counts: (
                FOR po IN purchase_orders
                COLLECT status = po.status WITH COUNT INTO count
                RETURN {status: status, count: count}
            ),
            undelivered: LENGTH(
                FOR po IN purchase_orders
                FILTER po.delivery_date == null
                RETURN po
            )
        }
        """
        try:
            cursor = self.db.db.aql.execute(aql)
            stats = next(cursor)
            self.logger.info(f"Total POs: {stats['total_pos']}")
            self.logger.info("Status Distribution:")
            for status in stats['status_counts']:
                self.logger.info(f"  {status['status']}: {status['count']}")
            self.logger.info(f"Undelivered Orders: {stats['undelivered']}")
        except Exception as e:
            self.logger.error(f"Error validating purchase orders: {str(e)}")

    def validate_inventory(self):
        """Validate inventory data."""
        self.logger.info("\n=== Inventory Validation ===")
        
        aql = """
        RETURN {
            total_inventory: LENGTH(inventory),
            low_stock: LENGTH(
                FOR inv IN inventory
                FILTER inv.quantity_on_hand <= inv.reorder_point
                RETURN inv
            ),
            total_transactions: LENGTH(inventory_transactions),
            transaction_types: (
                FOR t IN inventory_transactions
                COLLECT type = t.transaction_type WITH COUNT INTO count
                RETURN {type: type, count: count}
            )
        }
        """
        try:
            cursor = self.db.db.aql.execute(aql)
            stats = next(cursor)
            self.logger.info(f"Total Inventory Records: {stats['total_inventory']}")
            self.logger.info(f"Low Stock Items: {stats['low_stock']}")
            self.logger.info(f"Total Transactions: {stats['total_transactions']}")
            self.logger.info("Transaction Types:")
            for t_type in stats['transaction_types']:
                self.logger.info(f"  {t_type['type']}: {t_type['count']}")
        except Exception as e:
            self.logger.error(f"Error validating inventory: {str(e)}")

    def run_validation(self):
        """Run all validation checks."""
        self.logger.info("Starting Data Validation...")
        
        # Run all validations
        collection_counts = self.validate_all_collections()
        self.validate_relationships()
        self.validate_purchase_orders()
        self.validate_inventory()
        
        return collection_counts

if __name__ == "__main__":
    validator = DataValidator()
    validator.run_validation()