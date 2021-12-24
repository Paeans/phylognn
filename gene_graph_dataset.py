import sys
import time

import numpy as np
import torch

from gene_mat import gen_dataset_wt, gen_dataset_wb, gen_g2g_data, gen_m3g_data
from genome_graph import gen_graph, gen_g2g_graph, gen_g2b_graph

from torch_geometric.data import InMemoryDataset

# from multiprocessing import Pool

def save_dataset(gene_len, step_range, graph_num = None, fname = None):
    if graph_num == None:
        graph_num = 100
        
    if fname == None:
        fname = 'inv_' + str(gene_len) + '_' + str(step_range) + '.pt'
        
    gene = np.zeros((graph_num * step_range, 2, gene_len), dtype = np.int32) #[]
    label = np.zeros(graph_num * step_range, dtype = np.int32) #[]
    for step in range(0, step_range):
        s, o, t = gen_dataset_wt(gene_len, graph_num, step + 2, op_type = 2)
        gene[step * graph_num : (step + 1) * graph_num] = s[:, (0,-1)].astype(np.int32)
        label[step * graph_num : (step + 1) * graph_num] = step + 1 #inv_num = step + 1
        
#         g += [gen_graph(x, label = inv_num) for x in s]
    torch.save((gene, label), fname)

def save_g2b_dataset(gene_len, step_range, graph_num = None, fname = None):
    if graph_num == None:
        graph_num = 100
        
    if fname == None:
        fname = 'g2b_' + str(gene_len) + '_' + str(step_range) + '.pt'
        
    gene = np.zeros((graph_num * step_range, 2, gene_len), dtype = np.int32) #[]
    label = np.zeros(graph_num * step_range, dtype = np.int32) #[]
    node_label = np.zeros((graph_num * step_range, gene_len * 2), dtype = np.int32)
    
    for step in range(0, step_range):
        s, o, t, b = gen_dataset_wb(gene_len, graph_num, step + 2, op_type = 2)
        gene[step * graph_num : (step + 1) * graph_num] = s[:, (0,-1)].astype(np.int32)
        label[step * graph_num : (step + 1) * graph_num] = step + 1 #inv_num = step + 1
        
        node_label[step * graph_num : (step + 1) * graph_num][b.any(axis = 1)] = 1
#         g += [gen_graph(x, label = inv_num) for x in s]
    torch.save((gene, label, node_label), fname)

def save_g2g_dataset(gene_len, step, graph_num = None, fname = None):
    if graph_num == None:
        graph_num = 100
        
    if fname == None:
        fname = 'g2g_' + str(gene_len) + '_' + str(step) + '.pt'
    
    source = np.zeros((graph_num * step, 2, gene_len), dtype = np.int32) #[]
    target = np.zeros((graph_num * step, gene_len), dtype = np.int32) #[]
    
    for dist in range(0, step):
        s = gen_g2g_data(gene_len, graph_num, dist, op_type = 2)
        
        source[dist * graph_num : (dist + 1) * graph_num] = s[:, (0, -1)]
        target[dist * graph_num : (dist + 1) * graph_num] = s[:, 1]
    torch.save((source, target), fname)

def save_g3m_dataset(gene_len, step, graph_num = None, fname = None, mid_num = None):
    if graph_num == None:
        graph_num = 100
        
    if fname == None:
        fname = 'g3m_' + str(gene_len) + '_' + str(step) + '.pt'
        
    if mid_num == None:
        mid_num = 3
    
    source = np.zeros((graph_num * step, mid_num, gene_len), dtype = np.int32) 
    target = np.zeros((graph_num * step, gene_len), dtype = np.int32)
    
    for dist in range(0, step):
        # print(f'{time.ctime()} >> step: {dist:04d}, number: {graph_num:06d}, length: {gene_len:06d}', 
              # file = sys.stderr)
        
        m_seq, t_seq = gen_m3g_data(gene_len, graph_num, dist + 1, op_type = 2, mid_num = mid_num)
        source[dist * graph_num : (dist + 1) * graph_num] = m_seq
        target[dist * graph_num : (dist + 1) * graph_num] = t_seq.squeeze()
        
    torch.save((source, target), fname)

class GeneGraphDataset(InMemoryDataset):
    def __init__(self, root, gene_len, step_range, graph_num = 100):
#                  transform=None, pre_transform=None, pre_filter = None):
        self.gene_len = gene_len
        self.step_range = step_range
        self.graph_num = graph_num
        super().__init__(root + '_' + str(self.gene_len) + '_' 
                         + str(self.step_range) + '_' + str(self.graph_num), 
                         transform = None, 
                         pre_transform = None, 
                         pre_filter = None)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['inv_' + str(self.gene_len) +
                '_' + str(self.step_range) + '.pt']

    @property
    def processed_file_names(self):
        return ['data_' + str(self.gene_len) +
                '_' + str(self.step_range) + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        print('Generating...', file=sys.stderr)
        save_dataset(self.gene_len, self.step_range, 
                     graph_num = self.graph_num, 
                     fname = self.raw_dir + '/' + self.raw_file_names[0])
        pass

    def process(self):
        # Read data into huge `Data` list.
        filename = self.raw_dir + '/' + self.raw_file_names[0]
        gene_list, label = torch.load(filename) #, map_location=torch.device('cuda'))        
        
        data_list = [gen_graph(x, label = inv_num) for x, inv_num in zip(gene_list, label)]
        # with Pool(22) as p:
            # data_list = p.starmap(gen_graph, [(x, inv_num) for x, inv_num in zip(gene_list, label)])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class G2BraphDataset(InMemoryDataset):
    def __init__(self, root, gene_len, step_range, graph_num = 100):
#                  transform=None, pre_transform=None, pre_filter = None):
        self.gene_len = gene_len
        self.step_range = step_range
        self.graph_num = graph_num
        super().__init__(root + '_' + str(self.gene_len) + '_' 
                         + str(self.step_range) + '_' + str(self.graph_num), 
                         transform = None, 
                         pre_transform = None, 
                         pre_filter = None)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['b2raw_' + str(self.gene_len) +
                '_' + str(self.step_range) + '.pt']

    @property
    def processed_file_names(self):
        return ['b2dat_' + str(self.gene_len) +
                '_' + str(self.step_range) + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        print('Generating...', file=sys.stderr)
        save_g2b_dataset(self.gene_len, self.step_range, 
                     graph_num = self.graph_num, 
                     fname = self.raw_dir + '/' + self.raw_file_names[0])
        pass

    def process(self):
        # Read data into huge `Data` list.
        filename = self.raw_dir + '/' + self.raw_file_names[0]
        gene_list, label, node_label = torch.load(filename) #, map_location=torch.device('cuda'))        
        
        data_list = [gen_g2b_graph(x, label = inv_num, node_label = t_label) 
                     for x, inv_num, t_label in zip(gene_list, label, node_label)]
        # with Pool(22) as p:
            # data_list = p.starmap(gen_g2b_graph, 
                                  # [(x, inv_num, t_label) 
                                   # for x, inv_num, t_label in zip(gene_list, label, node_label)])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
class G2GraphDataset(InMemoryDataset):
    def __init__(self, root, gene_len, step_range, graph_num = 100):
#                  transform=None, pre_transform=None, pre_filter = None):
        self.gene_len = gene_len
        self.step_range = step_range
        self.graph_num = graph_num
        super().__init__(root + '_' + str(self.gene_len) + '_' 
                         + str(self.step_range) + '_' + str(self.graph_num), 
                         transform = None, 
                         pre_transform = None, 
                         pre_filter = None)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['g2raw_' + str(self.gene_len) +
                '_' + str(self.step_range) + '.pt']

    @property
    def processed_file_names(self):
        return ['g2dat_' + str(self.gene_len) +
                '_' + str(self.step_range) + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        print('Generating...', file=sys.stderr)
        save_g2g_dataset(self.gene_len, self.step_range, 
                     graph_num = self.graph_num, 
                     fname = self.raw_dir + '/' + self.raw_file_names[0])
        pass

    def process(self):
        # Read data into huge `Data` list.
        filename = self.raw_dir + '/' + self.raw_file_names[0]
        source, target = torch.load(filename) #, map_location=torch.device('cuda'))        
                
        data_list = [gen_g2g_graph(s, t) for s,t in zip(source, target)]
        # with Pool(22) as p:
            # data_list = p.starmap(gen_g2g_graph, [(s, t) for s, t in zip(source, target)])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
class G3MedianDataset(G2GraphDataset):
    def __init__(self, root, gene_len, step_range, graph_num = 100, mid_num = 3):
        self.mid_num = mid_num if mid_num >= 3 else 3
        super().__init__(root + '_' + str(self.mid_num), gene_len, step_range, graph_num)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['g3raw_' + str(self.gene_len) +
                '_' + str(self.step_range) + '.pt']

    @property
    def processed_file_names(self):
        return ['g3dat_' + str(self.gene_len) +
                '_' + str(self.step_range) + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        print('Generating...', file=sys.stderr)
        save_g3m_dataset(self.gene_len, self.step_range, 
                     graph_num = self.graph_num, 
                     fname = self.raw_dir + '/' + self.raw_file_names[0],
                     mid_num = self.mid_num)
        pass

    def process(self):
        super().process()