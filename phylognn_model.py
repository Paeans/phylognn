
import torch

import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear, Sigmoid
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn import GCNConv, PNAConv, BatchNorm # global_add_pool
from torch_geometric.nn import VGAE

class G2Braph(torch.nn.Module):
    def __init__(self, deg):
        super(G2Braph, self).__init__()

        self.node_emb = Embedding(10000, 50)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=50, out_channels=50,
                           aggregators=aggregators, scalers=scalers, 
                           deg=deg,
                           # edge_dim=50, 
                           towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(50))
            
        self.mlp = Sequential(Linear(50, 25), ReLU(),
                              Linear(25, 1), Sigmoid())

    def forward(self, x, edge_index, edge_attr, batch):
        
        x = self.node_emb(x.squeeze())
        
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            # x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            x = F.relu(batch_norm(conv(x, edge_index)))
        # x = global_add_pool(x, batch)        
        return self.mlp(x)

class G3Median(torch.nn.Module):
    def __init__(self, in_channels, out_channels, deg):
        super(G3Median, self).__init__()

        self.node_emb = Embedding(10000, 75)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.conv_mu = GCNConv(75, out_channels)
        self.conv_logstd = GCNConv(75, out_channels)
        
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           # edge_dim=50, 
                           towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            # conv = GCNConv(in_channels=75, out_channels=75)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

    def forward(self, x, edge_index): #, edge_attr, batch):
        
        x = self.node_emb(x.squeeze())

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = batch_norm(conv(x, edge_index)).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    
class G3Median_GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(G3Median_GCNConv, self).__init__()

        self.node_emb = Embedding(10000, 75)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.conv_mu = GCNConv(75, out_channels)
        self.conv_logstd = GCNConv(75, out_channels)
        
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            # conv = PNAConv(in_channels=75, out_channels=75,
            #                aggregators=aggregators, scalers=scalers, deg=deg,
            #                # edge_dim=50, 
            #                towers=5, pre_layers=1, post_layers=1,
            #                divide_input=False)
            conv = GCNConv(in_channels=75, out_channels=75)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

    def forward(self, x, edge_index): #, edge_attr, batch):
        
        x = self.node_emb(x.squeeze())

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = batch_norm(conv(x, edge_index)).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)