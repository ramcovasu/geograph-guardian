from arango import ArangoClient
import yaml
import os
from typing import Optional
from pathlib import Path

class ArangoDB:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ArangoDB, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.config = self._load_config()
        self.client = self._create_client()
        self.db = self._get_database()

    def _load_config(self) -> dict:
        """Load ArangoDB configuration from YAML file."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'arangodb.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config['development']

    def _create_client(self) -> ArangoClient:
        """Create ArangoDB client connection."""
        return ArangoClient(
            hosts=f"http://{self.config['host']}:{self.config['port']}"
        )

    def _get_database(self):
        """Get or create the database."""
        sys_db = self.client.db(
            '_system',
            username=self.config['username'],
            password=self.config['password']
        )

        if not sys_db.has_database(self.config['database']):
            sys_db.create_database(self.config['database'])

        return self.client.db(
            self.config['database'],
            username=self.config['username'],
            password=self.config['password']
        )

    def get_graph(self):
        """Get or create the graph."""
        graph_name = self.config['graph_name']
        if not self.db.has_graph(graph_name):
            self.db.create_graph(graph_name)
        return self.db.graph(graph_name)

    def create_collections(self):
        """Create all collections defined in config."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'arangodb.yaml'
        with open(config_path, 'r') as file:
            collections_config = yaml.safe_load(file)['collections']

        # Create vertex collections
        for vertex in collections_config['vertices']:
            collection_name = vertex['name']
            if not self.db.has_collection(collection_name):
                self.db.create_collection(collection_name)

        # Create edge collections
        for edge in collections_config['edges']:
            collection_name = edge['name']
            if not self.db.has_collection(collection_name):
                self.db.create_collection(collection_name, edge=True)

    def get_collection(self, name: str):
        """Get a collection by name."""
        return self.db.collection(name)

    def close(self):
        """Close the database connection."""
        if self.client:
            self.client.close()