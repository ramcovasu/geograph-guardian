import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import pandas as pd
from .plot_utils import prepare_data_for_plot

class ChartGenerator:
    def __init__(self):
        self.default_template = "plotly_dark"  # Better for data visibility
        self.color_sequence = px.colors.qualitative.D3  # More professional color scheme
        self.default_height = 500  # Increased height
        self.default_width = 900   # Increased width

    def _validate_chart_config(self, config: Dict[str, Any]) -> None:
        """Validate chart configuration structure."""
        required_fields = [
            'config',
            'config.data',
            'config.layout',
            'config.data.x',
            'config.data.y'
        ]
        
        for field in required_fields:
            if field.count('.') == 0:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")
            else:
                parts = field.split('.')
                current = config
                for part in parts:
                    if part not in current:
                        raise ValueError(f"Missing required field: {field}")
                    current = current[part]

    def should_create_chart(self, data: List[Dict[str, Any]]) -> bool:
        """Determine if data is suitable for visualization."""
        if not data or len(data) == 0:
            print("No data available for chart")
            return False
            
        # Check if we have numeric data
        sample = data[0]
        print(f"Sample data: {sample}")
        print(f"Types: {[(k, type(v)) for k, v in sample.items()]}")
        
        numeric_fields = [k for k, v in sample.items() 
                        if isinstance(v, (int, float)) and k not in ['_id', '_key']]
        print(f"Numeric fields found: {numeric_fields}")
        
        # Need at least one numeric field and enough data points
        return len(numeric_fields) > 0 and len(data) > 1
    
    def prepare_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare data for visualization with validation."""
        try:
            if not data:
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            # Handle missing values
            df = df.fillna(0)
            return df
        except Exception:
            return pd.DataFrame()

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data for visualization."""
        if df.empty:
            return False
            
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) == 0:
            return False
            
        # Ensure we have enough data points for visualization
        if len(df) < 2:
            return False
            
        return True
    
    def create_chart(self, chart_config: Dict[str, Any], chart_type: str = None) -> Optional[go.Figure]:
        """Create an enhanced chart with better styling and readability."""
        try:
            if not chart_config:
                print("No chart configuration provided")
                return None
                
            # Validate configuration and data
            self._validate_chart_config(chart_config)
            data = chart_config['config']['data']
            layout = chart_config['config']['layout']
            
            # Debug data
            print(f"Creating chart with data: {data}")
            print(f"Chart type: {chart_type or chart_config.get('chart_type', 'bar')}")
            
            if not data.get('x') or not data.get('y'):
                print("Missing x or y data for chart")
                return None

            chart_type = chart_type or chart_config.get('chart_type', 'bar')
            
            # Create figure based on chart type
            if chart_type == "bar":
                fig = self._create_enhanced_bar_chart(data, layout)
            elif chart_type == "line":
                fig = self._create_enhanced_line_chart(data, layout)
            elif chart_type == "scatter":
                fig = self._create_enhanced_scatter_chart(data, layout)
            elif chart_type == "pie":
                fig = self._create_enhanced_pie_chart(data, layout)
            else:
                print(f"Unsupported chart type: {chart_type}")
                return None

            if fig is None:
                print("Failed to create chart")
                return None

            # Apply enhanced layout
            self._apply_enhanced_layout(fig, layout)
            return fig
                
        except Exception as e:
            print(f"Error creating chart: {str(e)}")
            return None

    def _create_enhanced_bar_chart(self, data: Dict[str, List], layout: Dict[str, Any]) -> Optional[go.Figure]:
        """Create an enhanced bar chart with better readability."""
        try:
            return go.Figure(data=[
                go.Bar(
                    x=data['x'],
                    y=data['y'],
                    text=data['y'],
                    textposition='auto',
                    marker_color=self.color_sequence,
                    hovertemplate="<b>%{x}</b><br>" +
                                "Value: %{y:,.2f}<br>" +
                                "<extra></extra>"
                )
            ])
        except Exception:
            return None

    def _create_enhanced_line_chart(self, data: Dict[str, List], layout: Dict[str, Any]) -> Optional[go.Figure]:
        """Create an enhanced line chart with better readability."""
        try:
            return go.Figure(data=[
                go.Scatter(
                    x=data['x'],
                    y=data['y'],
                    mode='lines+markers+text',
                    text=data['y'],
                    textposition='top center',
                    marker=dict(size=8),
                    line=dict(width=3),
                    hovertemplate="<b>%{x}</b><br>" +
                                "Value: %{y:,.2f}<br>" +
                                "<extra></extra>"
                )
            ])
        except Exception:
            return None

    def _create_enhanced_pie_chart(self, data: Dict[str, List], layout: Dict[str, Any]) -> Optional[go.Figure]:
        """Create an enhanced pie chart with better readability."""
        try:
            return go.Figure(data=[
                go.Pie(
                    values=data['y'],
                    labels=data['x'],
                    marker_colors=self.color_sequence,
                    textposition='inside',
                    textinfo='label+percent',
                    insidetextorientation='radial',
                    hovertemplate="<b>%{label}</b><br>" +
                                "Value: %{value:,.2f}<br>" +
                                "Percentage: %{percent:.1%}<br>" +
                                "<extra></extra>"
                )
            ])
        except Exception:
            return None

    def _apply_enhanced_layout(self, fig: go.Figure, layout: Dict[str, Any]) -> None:
        """Apply enhanced layout settings with better styling."""
        try:
            fig.update_layout(
                title=dict(
                    text=layout.get('title', ''),
                    font=dict(size=24),
                    x=0.5,
                    xanchor='center'
                ),
                template=self.default_template,
                height=self.default_height,
                width=self.default_width,
                margin=dict(l=60, r=40, t=80, b=60),
                showlegend=layout.get('showlegend', True),
                legend=dict(
                    x=1.02,
                    y=1,
                    bgcolor='rgba(255, 255, 255, 0.1)',
                    bordercolor='rgba(255, 255, 255, 0.2)'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14)
            )

            # Enhanced grid styling
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                title_font=dict(size=14),
                tickfont=dict(size=12)
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                title_font=dict(size=14),
                tickfont=dict(size=12)
            )

        except Exception:
            # Fallback to basic layout if enhanced fails
            fig.update_layout(
                title=layout.get('title', ''),
                template=self.default_template
            )
   