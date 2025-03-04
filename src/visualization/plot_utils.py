import pandas as pd
from typing import Dict, Any, Union
import numpy as np

def prepare_data_for_plot(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert ArangoDB query results to pandas DataFrame suitable for plotting.
    
    Args:
        data: Dictionary containing query results
        
    Returns:
        DataFrame prepared for visualization
    """
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        return pd.DataFrame([data])
    else:
        raise ValueError("Unsupported data format")

def format_numeric_values(value: Union[float, int]) -> str:
    """Format numeric values for display in charts."""
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}"

def calculate_chart_dimensions(data_length: int) -> Dict[str, int]:
    """Calculate appropriate chart dimensions based on data size."""
    base_height = 400
    base_width = 800
    
    if data_length <= 5:
        return {"height": base_height, "width": base_width}
    elif data_length <= 10:
        return {"height": base_height, "width": base_width * 1.2}
    else:
        return {"height": base_height * 1.5, "width": base_width * 1.5}

def preprocess_time_series(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Preprocess time series data for visualization."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    return df.sort_values(date_column)

def aggregate_data(df: pd.DataFrame, group_by: str, agg_column: str, 
                  agg_func: str = 'sum') -> pd.DataFrame:
    """Aggregate data for visualization."""
    return df.groupby(group_by)[agg_column].agg(agg_func).reset_index()

def calculate_moving_average(df: pd.DataFrame, value_column: str, 
                           window: int = 3) -> pd.Series:
    """Calculate moving average for trend visualization."""
    return df[value_column].rolling(window=window, min_periods=1).mean()