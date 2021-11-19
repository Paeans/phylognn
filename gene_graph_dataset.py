import sys

import numpy as np
import torch

from gene_mat import gen_dataset_wt
from genome_graph import gen_graph

from torch_geometric.data import InMemoryDataset

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
    
class GeneGraphDataset(InMemoryDataset):
    def __init__(self, root, gene_len, step_range, graph_num = 100):
#                  transform=None, pre_transform=None, pre_filter = None):
        self.gene_len = gene_len
        self.step_range = step_range
        self.graph_num = graph_num
        super().__init__(root, transform = None, 
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
        gene_list, label = torch.load(filename, map_location=torch.device('cuda'))        
        
        data_list = [gen_graph(x, label = inv_num) for x, inv_num in zip(gene_list, label)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])