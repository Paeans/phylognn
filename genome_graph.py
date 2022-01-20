import numpy as np

import torch
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import (negative_sampling, 
                                   add_self_loops, 
                                   to_dense_adj, 
                                   to_undirected)

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

def gen_g2b_graph(genome, label = None, node_label = None):
    graph_data = gen_graph(genome, label)
    graph_data.node_label = torch.tensor(node_label, dtype = torch.float)
        
    return graph_data

def gen_graph(genome, label = None): # label number
    gene_adj = [encodeAdj(gene) for gene in genome]
    num_edges = np.sum([len(g) - 3 for g in gene_adj])
    num_genes = len(genome)
    
    graph_adj = np.zeros((2, num_edges), dtype = np.int32)
    edge_attr = np.zeros((num_edges, num_genes), dtype = np.int32)
    
    L, g_index = 0, 0
    for gene in gene_adj:
        gene_s = gene[1:-2]
        gene_e = gene[2:-1]
        l = len(gene_s)
        
        graph_adj[0, L : L + l] = gene_s
        graph_adj[1, L : L + l] = gene_e
        
        edge_attr[L : L + l, g_index] = 1
        
        L += l
        g_index += 1
        
    node_num = np.max(gene_adj) + 1
    # node_x = np.zeros((node_num, 2), dtype = np.int32)
    # node_x[np.arange(node_num) % 2 == 0, 0] = 1
    # node_x[np.arange(node_num) % 2 == 1, 1] = 1
    
    edge_index = torch.tensor(graph_adj, dtype = torch.long)    
    # edge_index = to_undirected(edge_index)
    
    # node_x = to_dense_adj(edge_index).to(torch.long).squeeze()    
    # node_x[node_x > 0] = 1
    
    node_x = torch.tensor(np.arange(node_num), dtype = torch.long)
    
    graph_data = Data(x = node_x, 
                      edge_index = edge_index, # torch.tensor(graph_adj, dtype = torch.long), 
                      edge_attr = torch.tensor(edge_attr),
                      dtype = torch.long, num_nodes = node_num)
    graph_data.y = torch.tensor([label if label else 0], dtype = torch.long)
        
    return graph_data
        
def gen_g2g_graph_old(genome, target, nf_base = 0):
    gene_adj = [encodeAdj(gene) for gene in genome]
    num_edges = np.sum([len(g) - 3 for g in gene_adj])
    num_genes = len(genome)
    
    graph_adj = np.zeros((2, num_edges), dtype = np.int32)
    edge_attr = np.zeros((num_edges, num_genes), dtype = np.int32)
    
    L, g_index = 0, 0
    for gene in gene_adj:
        gene_s = gene[1:-2]
        gene_e = gene[2:-1]
        l = len(gene_s)
        
        graph_adj[0, L : L + l] = gene_s
        graph_adj[1, L : L + l] = gene_e
        
        edge_attr[L : L + l, g_index] = 1
        
        L += l
        g_index += 1
        
    node_num = np.max(gene_adj) + 1
    # node_x = np.zeros((node_num, 2), dtype = np.int32)
    # node_x[np.arange(node_num) % 2 == 0, 0] = 1
    # node_x[np.arange(node_num) % 2 == 1, 1] = 1
    node_x = np.arange(node_num) # + nf_base
    
    graph_data = Data(x = torch.tensor(node_x, dtype = torch.long), 
                      edge_index = torch.tensor(graph_adj, dtype = torch.long), 
                      edge_attr = torch.tensor(edge_attr), num_nodes = node_num)
                      # dtype = torch.float, num_nodes = node_num)
    target_graph = gen_graph([target], 0)
    # graph_data.pos_edge_label_index = target_graph.edge_index
    graph_data.pos_edge_label_index, _ = add_self_loops(to_undirected(target_graph.edge_index))
    graph_data.neg_edge_label_index = negative_sampling(graph_data.pos_edge_label_index, 
                                                        node_num,
                                                        node_num**2)
    
    return graph_data    
        
def gen_pos_neg_edge(edge_index, node_num):
    adj_matrix = to_dense_adj(to_undirected(edge_index), 
                              max_num_nodes = node_num).squeeze()
    assert node_num % 2 == 0
    n = node_num // 2
    pos_edge_index = np.zeros((2, n - 1))
    neg_edge_index = np.zeros((2, n * (node_num - 2) - (n - 1)))
    
    p_index, n_index = 0, 0
    for i in range(0, node_num, 2):
        for j in range(i + 2, node_num):
            if adj_matrix[i, j] == 1:
                pos_edge_index[:, p_index] = [i, j]
                p_index += 1
            else:
                neg_edge_index[:, n_index] = [i, j]
                n_index += 1
                
            if adj_matrix[i + 1, j] == 1:
                pos_edge_index[:, p_index] = [i + 1, j]
                p_index += 1
            else:
                neg_edge_index[:, n_index] = [i + 1, j]
                n_index += 1
                
    return (torch.tensor(pos_edge_index, dtype=torch.long), 
            torch.tensor(neg_edge_index, dtype=torch.long))
    
def gen_g2g_graph(genome, target, nf_base = 0):
    gene_adj = [encodeAdj(gene) for gene in genome]
    num_edges = np.sum([len(g) - 3 for g in gene_adj])
    num_genes = len(genome)
    
    graph_adj = np.zeros((2, num_edges), dtype = np.int32)
    edge_attr = np.zeros((num_edges, num_genes), dtype = np.int32)
    
    L, g_index = 0, 0
    for gene in gene_adj:
        gene_s = gene[1:-2]
        gene_e = gene[2:-1]
        l = len(gene_s)
        
        graph_adj[0, L : L + l] = gene_s
        graph_adj[1, L : L + l] = gene_e
        
        edge_attr[L : L + l, g_index] = 1
        
        L += l
        g_index += 1
        
    node_num = np.max(gene_adj) + 1
    # node_x = np.zeros((node_num, 2), dtype = np.int32)
    # node_x[np.arange(node_num) % 2 == 0, 0] = 1
    # node_x[np.arange(node_num) % 2 == 1, 1] = 1
    node_x = np.arange(node_num) # + nf_base
    
    graph_data = Data(x = torch.tensor(node_x, dtype = torch.long), 
                      edge_index = torch.tensor(graph_adj, dtype = torch.long), 
                      edge_attr = torch.tensor(edge_attr), num_nodes = node_num)
                      # dtype = torch.float, num_nodes = node_num)
    target_graph = gen_graph([target], 0)
    # graph_data.pos_edge_label_index = target_graph.edge_index
    # graph_data.pos_edge_label_index, _ = add_self_loops(to_undirected(target_graph.edge_index))
    # graph_data.neg_edge_label_index = negative_sampling(graph_data.pos_edge_label_index, 
    #                                                     node_num,
    #                                                     node_num**2)
    graph_data.pos_edge_label_index, graph_data.neg_edge_label_index = gen_pos_neg_edge(target_graph.edge_index, node_num)
    
    return graph_data 

def gen_single_graph(genome):
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

# def gen_multi_graph(*genomes):
#     graphs = []
#     for genome in genomes:
#         graph_data = gen_single_graph(genome)
#         graphs.append(graph_data)
#     return graphs

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
    arc_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    style_list = ['--', '-.', ':', '-']
    for i, graph in enumerate(pgraph):
        arc = arc_list[i] if i <=7 else arc_list[np.random.rand(1)[0]]
#         if i <= 7:
#             arc = arc_list[i]
#             connectionstyle = 'arc3,rad=' + str(i)
#         else:
#             connectionstyle = 'arc3,rad=' + str(np.random.rand(1)[0])
        connectionstyle = 'arc3,rad=' + str(arc)
        arcs = nx.draw_networkx_edges(graph, pos=pos, 
                                      connectionstyle=connectionstyle, #'arc3,rad=0.3',
                                      edge_color = color_list[i%7],
                                      width = 2,
                                     style = style_list[i%4])
#         for arc in arcs:  # change alpha values of arcs
#             arc.set_alpha(alpha_list[i%7])
    plt.show()
    

