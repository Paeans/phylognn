import numpy as np

import torch
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

import matplotlib.pyplot as plt

from genome_file import encodeAdj


# def gen_graph(*genomes):
#     graph_adj = np.empty((2,0), dtype = np.int32)
#     max_node_index = []
#     for genome in genomes:
#         gene_adj = encodeAdj(genome)        
#         graph_adj = np.concatenate(
#             (graph_adj, np.stack((gene_adj[1:-2], gene_adj[2:-1]))), 
#             axis = 1)
#         max_node_index.append(gene_adj.max())
        
#     node_num = max(max_node_index) + 1
#     node_x = np.zeros((node_num, 2), dtype = np.int32)
#     node_x[np.arange(node_num) % 2 == 0, 0] = 1
#     node_x[np.arange(node_num) % 2 == 1, 1] = 1
    
#     graph_data = Data(x = node_x, edge_index = torch.tensor(graph_adj), 
#                       dtype = torch.long, num_nodes = node_num)
        
#     return graph_data

# def plot_single_network(data, figsize = (120, 90)):
#     pgraph = to_networkx(data)
#     node_labels = np.arange(len(data.x))
    
#     plt.figure(figsize = figsize)

#     pos = nx.spring_layout(pgraph)
#     nx.draw_networkx_nodes(pgraph, pos=pos, 
#                            cmap = plt.get_cmap('Set1'),
#                            node_size = 80)
#     nx.draw_networkx_labels(pgraph, pos=pos, font_size=10)
#     arcs = nx.draw_networkx_edges(pgraph, pos=pos, 
#                                   connectionstyle='arc3,rad=0.3',
#                                   edge_cmap = plt.get_cmap('Set1'),
#                                   width = 2)
#     for arc in arcs:  # change alpha values of arcs
#         arc.set_alpha(0.3)
#     plt.show()

def gen_single_graph(*genome):
    graph_adj = np.empty((2,0), dtype = np.int32)
    max_node_index = []

    for gene in genome:
        gene_adj = encodeAdj(gene)        
        graph_adj = np.concatenate(
            (graph_adj, np.stack((gene_adj[1:-2], gene_adj[2:-1]))), 
            axis = 1)
        max_node_index.append(gene_adj.max())
        
    node_num = max(max_node_index) + 1
    node_x = np.zeros((node_num, 2), dtype = np.int32)
    node_x[np.arange(node_num) % 2 == 0, 0] = 1
    node_x[np.arange(node_num) % 2 == 1, 1] = 1
    
    graph_data = Data(x = node_x, edge_index = torch.tensor(graph_adj), 
                      dtype = torch.long, num_nodes = node_num)
        
    return graph_data

def gen_multi_graph(*genomes):
    graphs = []
    for genome in genomes:
        graph_data = gen_single_graph(genome)
        graphs.append(graph_data)
    return graphs

def plot_multi_graph(data, figsize = (120, 90), pos = None):
    pgraph = [to_networkx(d) for d in data]
    # node_labels = np.arange(len(data.x))
    
    plt.figure(figsize = figsize)

    pg = pgraph[0]
    if pos == None:
        pos = nx.spring_layout(pg)
    nx.draw_networkx_nodes(pg, pos=pos, 
                           cmap = plt.get_cmap('Set1'),
                           node_size = 80)
    nx.draw_networkx_labels(pg, pos=pos, font_size=10)
    
    alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    arc_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for i, graph in enumerate(pgraph):
        if i <= 7:
            connectionstyle = 'arc3,rad=' + str(i)
        else:
            connectionstyle = 'arc3,rad=' + str(np.random.rand(1)[0])
        arcs = nx.draw_networkx_edges(graph, pos=pos, 
                                      connectionstyle=connectionstyle, #'arc3,rad=0.3',
                                      edge_color = color_list[i%7],
                                      width = 2)
#         for arc in arcs:  # change alpha values of arcs
#             arc.set_alpha(alpha_list[i%7])
    plt.show()