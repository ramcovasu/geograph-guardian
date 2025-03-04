import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """Load all configuration files."""
        config_dir = Path(__file__).parent.parent.parent / 'config'
        self._config = {}
        
        # Load ArangoDB config
        with open(config_dir / 'arangodb.yaml', 'r') as f:
            self._config['arangodb'] = yaml.safe_load(f)

    def get_db_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self._config['arangodb']['development']

    def get_collections_config(self) -> Dict[str, Any]:
        """Get collections configuration."""
        return self._config['arangodb']['collections']

    def get_vertex_collections(self) -> list:
        """Get list of vertex collections."""
        return self._config['arangodb']['collections']['vertices']

    def get_edge_collections(self) -> list:
        """Get list of edge collections."""
        return self._config['arangodb']['collections']['edges']