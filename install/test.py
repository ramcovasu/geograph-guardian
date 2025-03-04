import cudf
import cugraph
import numpy as np

# Create sample supply chain data
edges = cudf.DataFrame({
    'supplier': ['A', 'B', 'C'],
    'customer': ['B', 'C', 'A'],
    'risk_score': [0.3, 0.5, 0.2]
})

print("Edge Data:")
print(edges)

# Create graph - let's try with directed=True
G = cugraph.Graph(directed=True)
G.from_cudf_edgelist(edges, 
                     source='supplier', 
                     destination='customer', 
                     edge_attr='risk_score')

print("\nSupply Chain Graph Created Successfully!")
print(f"Number of nodes: {G.number_of_vertices()}")
print(f"Number of connections: {G.number_of_edges()}")

# Let's see the edge list from the graph
print("\nGraph Edges:")
print(G.view_edge_list())