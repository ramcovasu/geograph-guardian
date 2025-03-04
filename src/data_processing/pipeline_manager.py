import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from src.utils.logging import Logger

class DataPipelineManager:
    def __init__(self):
        self.logger = Logger().get_logger()
        self.base_path = Path(__file__).parent.parent.parent
        self.raw_path = self.base_path / 'data' / 'raw'
        self.processed_path = self.base_path / 'data' / 'processed'
        
        # Ensure processed directory exists
        self.processed_path.mkdir(exist_ok=True)

    def process_file(self, filename: str, processing_config: Dict[str, Any]) -> bool:
        """
        Process a single file according to its configuration.
        
        Args:
            filename: Name of the file to process
            processing_config: Dictionary containing processing instructions
        """
        try:
            # Read raw data
            raw_file = self.raw_path / filename
            df = pd.read_csv(raw_file)
            self.logger.info(f"Read {filename} with {len(df)} rows")

            # Apply processing steps
            df = self.apply_processing_steps(df, processing_config)
            
            # Save processed data
            processed_file = self.processed_path / filename
            df.to_csv(processed_file, index=False)
            self.logger.info(f"Saved processed {filename}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error processing {filename}: {str(e)}")
            return False

    def apply_processing_steps(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply processing steps defined in config to the dataframe."""
        try:
            # Handle missing values
            if config.get('handle_missing'):
                for col, strategy in config['handle_missing'].items():
                    if strategy == 'drop':
                        df = df.dropna(subset=[col])
                    elif isinstance(strategy, (str, int, float)):
                        df[col] = df[col].fillna(strategy)

            # String cleaning
            if config.get('string_columns'):
                for col in config['string_columns']:
                    if col in df.columns:
                        df[col] = df[col].str.strip().str.upper()

            # Data type conversions
            if config.get('dtype_conversions'):
                for col, dtype in config['dtype_conversions'].items():
                    if col in df.columns:
                        df[col] = df[col].astype(dtype)

            # Custom transformations
            if config.get('custom_transformations'):
                for transform in config['custom_transformations']:
                    df = transform(df)

            return df
        except Exception as e:
            self.logger.error(f"Error in processing steps: {str(e)}")
            raise

def get_processing_configs() -> Dict[str, Dict[str, Any]]:
    """Define processing configurations for each file."""
    return {
        'products.csv': {
            'handle_missing': {
                'product_id': 'drop',
                'description': ''
            },
            'string_columns': ['product_id', 'product_name', 'product_category'],
            'dtype_conversions': {
                'product_id': str,
                'product_category': str
            }
        },
        
        'product_parts.csv': {
            'handle_missing': {
                'part_id': 'drop',
                'part_name': 'drop',
                'criticality_level': 'MEDIUM'
            },
            'string_columns': ['part_id', 'part_name', 'part_category', 'criticality_level'],
            'dtype_conversions': {
                'part_id': str,
                'criticality_level': str
            }
        },
        
        'supplier_master.csv': {
            'handle_missing': {
                'supplier_id': 'drop',
                'supplier_rating': 'C'
            },
            'string_columns': ['supplier_id', 'supplier_name', 'country', 'region', 'supplier_rating'],
            'dtype_conversions': {
                'supplier_id': str,
                'established_date': 'datetime64[ns]'
            }
        },
        
        'supplier_parts.csv': {
            'handle_missing': {
                'supplier_id': 'drop',
                'part_id': 'drop',
                'unit_cost': 0.0,
                'lead_time_days': 30,
                'minimum_order_qty': 1
            },
            'dtype_conversions': {
                'supplier_id': str,
                'part_id': str,
                'is_primary_supplier': bool,
                'unit_cost': float,
                'lead_time_days': int,
                'minimum_order_qty': int
            }
        },
        
        'purchase_orders.csv': {
            'handle_missing': {
                'po_id': 'drop',           # Drop if missing (but we don't have any missing)
                'supplier_id': 'drop',
                'part_id': 'drop',
                'quantity': 0,             # Default to 0 if missing
                'unit_price': 0.0,         # Default to 0.0 if missing
                'status': 'IN_TRANSIT'     # Default status if missing
            },
            'string_columns': ['po_id', 'supplier_id', 'part_id', 'status'],
            'dtype_conversions': {
                'po_id': str,
                'supplier_id': str,
                'part_id': str,
                'order_date': 'datetime64[ns]',
                'promised_date': 'datetime64[ns]',
                # Don't include delivery_date in dtype conversions
                'quantity': int,
                'unit_price': float
            },
            'custom_transformations': [
                # Leave delivery_date as is - NULL values are valid for undelivered orders
                lambda df: df
            ]
        },
        
        'supplier_risk_factors.csv': {
            'handle_missing': {
                'supplier_id': 'drop',
                'risk_category': 'UNKNOWN',
                'risk_score': 0.5,
                'comments': ''
            },
            'string_columns': ['supplier_id', 'risk_category', 'comments'],
            'dtype_conversions': {
                'supplier_id': str,
                'risk_score': float,
                'assessment_date': 'datetime64[ns]'
            }
        },
        
        'part_dependencies.csv': {
            'handle_missing': {
                'part_id': 'drop',
                'dependent_part_id': 'drop',
                'dependency_type': 'UNKNOWN'
            },
            'string_columns': ['part_id', 'dependent_part_id', 'dependency_type'],
            'dtype_conversions': {
                'part_id': str,
                'dependent_part_id': str
            }
        },
        
        'inventory_current.csv': {
            'handle_missing': {
                'part_id': 'drop',
                'warehouse_location': 'UNKNOWN',
                'quantity_on_hand': 0,
                'reorder_point': 0,
                'safety_stock': 0
            },
            'string_columns': ['part_id', 'warehouse_location', 'unit_of_measure'],
            'dtype_conversions': {
                'part_id': str,
                'quantity_on_hand': int,
                'reorder_point': int,
                'safety_stock': int
            }
        },
        
        'inventory_transactions.csv': {
            'handle_missing': {
                'transaction_id': 'drop',
                'part_id': 'drop',
                'transaction_type': 'UNKNOWN',
                'po_id': None,
                'reference_doc': ''
            },
            'string_columns': ['transaction_id', 'part_id', 'transaction_type', 'po_id', 'reference_doc'],
            'dtype_conversions': {
                'transaction_id': str,
                'part_id': str,
                'transaction_date': 'datetime64[ns]',
                'quantity': int
            }
        }
    }
def process_all_files():
    """Process all files using the pipeline manager."""
    pipeline = DataPipelineManager()
    configs = get_processing_configs()
    
    results = {}
    for filename, config in configs.items():
        results[filename] = pipeline.process_file(filename, config)
    
    return results

if __name__ == "__main__":
    results = process_all_files()
    for filename, success in results.items():
        print(f"{filename}: {'Success' if success else 'Failed'}")