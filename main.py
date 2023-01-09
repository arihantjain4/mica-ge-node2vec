# run with python 3.9

import anndata
from pynndescent import NNDescent
import networkx as nx
import torch
import torch_geometric
from torch_geometric.nn import Node2Vec
import os

def line():
    print("\n\n", ('_' * os.get_terminal_size().columns-1), "\n\n")

# read data file
annDataObj = anndata.read_h5ad("./pbmc3k_preprocessed.h5ad")
frame = annDataObj.to_df()
print(frame)
line()



# create networkx graph, then convert to torch_geometric 'Data' data type
knn_indices, knn_dists = NNDescent(frame.to_numpy(), n_neighbors=80, pruning_degree_multiplier=3.0, n_jobs=1).neighbor_graph # replace default constants with args
knn_graph = nx.Graph()
for c_index, cell in enumerate(knn_indices):
    knn_graph.add_node(cell[0])
    for n_index, neighbor in enumerate(cell):
        if n_index == 0:
            continue
        knn_graph.add_node(neighbor)
        knn_graph.add_edge(cell[0], neighbor, MI=knn_dists[c_index][n_index])

graph_data = torch_geometric.utils.from_networkx(knn_graph)


# torch_geometric Node2Vec
pyg_model = Node2Vec(graph_data.edge_index, embedding_dim=20, walk_length=100, context_size=20, walks_per_node=120, p=0.5, q=0.5, sparse=True).to("cuda" if torch.cuda.is_available() else "cpu") # replace default constants with args
pyg_forward = pyg_model.forward().detach().numpy()


# manually write to emb_file
emb_file = 'knn_graph_emb_node2vec_20.txt'
with open(emb_file, "w") as f:
    f.write(str(pyg_forward.shape[0]) + " " + str(pyg_forward.shape[1]))
    f.write("\n")
    for w in pyg_forward:
        f.write(", ".join(str(e) for e in w))
        f.write("\n")
