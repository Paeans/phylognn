
import torch

import torch.nn.functional as F
from torch.nn import ModuleList, Embedding, Dropout, MaxPool1d
from torch.nn import Sequential, ReLU, Linear, Sigmoid
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn import GCNConv, PNAConv, BatchNorm
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import VGAE

from torch_geometric.utils import to_dense_adj

EPS = 1e-15

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
    
class G2Braph_GCNConv(torch.nn.Module):
    def __init__(self):
        super(G2Braph_GCNConv, self).__init__()

        self.node_emb = Embedding(10000, 25)

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(2):
            conv = GCNConv(in_channels=25, out_channels=25)
            
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(25))
            
        self.mlp = Sequential(Linear(25, 12), ReLU(),
                              Linear(12, 1), Sigmoid())

    def forward(self, x, edge_index):
        
        x = self.node_emb(x.squeeze())
        
        for conv, batch_norm in zip(self.convs, self.batch_norms):
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

        self.node_emb = Embedding(10000, in_channels)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)
        
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = GCNConv(in_channels=in_channels, out_channels=in_channels)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(in_channels))

    def forward(self, x, edge_index): #, edge_attr, batch):
        
        x = self.node_emb(x.squeeze())

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = batch_norm(conv(x, edge_index)).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    
class G3Median_VGAE(VGAE):
    def __init__(self, encoder, decoder = None):
        super(G3Median_VGAE, self).__init__(encoder, decoder)
        
    def pred(self, z, pos_edge_index, neg_edge_index = None):
        pos_y = z.new_ones(pos_edge_index.size(1))
        
        if neg_edge_index == None:        
            neg_y = z.new_zeros(z.size(0)**2 - pos_edge_index.size(1)) # neg_edge_index.size(1))
        else:
            neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
        
        if neg_edge_index == None:
            pred = self.decoder.forward_all(z)
            adj = to_dense_adj(pos_edge_index).squeeze()
            pos_pred, neg_pred = pred[adj == 1], pred[adj == 0]
        else:
            pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
            neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
            
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        
        return y, pred
    
#     def recon_loss(self, z, pos_edge_index):
#         y, pred = to_dense_adj(pos_edge_index).squeeze(), self.decoder.forward_all(z)
        
#         pos_loss = -torch.log(pred[y == 1] + EPS).mean()
#         neg_loss = -torch.log(1 - pred[y == 0] + EPS).mean()

#         return pos_loss + neg_loss

    def recon_loss_wt(self, z, pos_edge_index, neg_edge_index=None, pwt=1, nwt=1):
        
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pwt * pos_loss + nwt * neg_loss
        
    
class G2Dist_GCNConv(torch.nn.Module):
    def __init__(self):
        super(G2Dist_GCNConv, self).__init__()

        self.node_emb = Embedding(10000, 2)
        
        self.convs = ModuleList()
        self.pools = ModuleList()
        self.batch_norms = ModuleList()
        # for _ in range(2):            
        conv = GCNConv(in_channels=80, out_channels=100)
        self.convs.append(conv)
        self.pools.append(MaxPool1d(4, stride = 2))
        self.batch_norms.append(BatchNorm(49))
        
        conv = GCNConv(in_channels=49, out_channels=49)
        self.convs.append(conv)
        self.pools.append(MaxPool1d(3, stride = 2))
        self.batch_norms.append(BatchNorm(24))
        
        conv = GCNConv(in_channels=24, out_channels=24)
        self.convs.append(conv)
        self.pools.append(MaxPool1d(2))
        self.batch_norms.append(BatchNorm(12))
        
        conv = GCNConv(in_channels=12, out_channels=12)
        self.convs.append(conv)
        self.pools.append(MaxPool1d(2, stride = 1))
        self.batch_norms.append(BatchNorm(11))
            
        self.mlp = Sequential(#Linear(25, 12), ReLU(), Dropout(0.2),
                              Linear(2, 1))#, ReLU())
        self.lin = Sequential(Linear(440, 200), MaxPool1d(2), ReLU(), Dropout(0.2),
                              Linear(100, 50), MaxPool1d(2), ReLU(), Dropout(0.2),
                              Linear(25, 20))
    
    def forward(self, x, edge_index, batch):        
        x = self.node_emb(x.squeeze()).view(-1, 80)
        # x = x.to(torch.float)
        # for conv in self.pre:
        #     x = F.dropout(conv(x, edge_index), 0.2)
        
        for conv, pool, batch_norm in zip(self.convs, self.pools, self.batch_norms):
            x = F.dropout(conv(x, edge_index), 0.2)
            x = pool(x)
            x = F.relu(F.dropout(batch_norm(x), 0.2)) 
            # x = F.relu(F.dropout(x, 0.2)) 
            # x = F.relu(batch_norm(conv(x, edge_index)))
        # x = self.mlp(x)       
        # return global_max_pool(x, batch)
        return self.lin(x.view(-1, 440))
    
class G2Dist_GCNConv_Small(torch.nn.Module):
    def __init__(self):
        super(G2Dist_GCNConv_Small, self).__init__()

        self.node_emb = Embedding(10000, 2)
        
        self.convs = ModuleList()
        self.pools = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(2):            
            conv = GCNConv(in_channels=80, out_channels=80)
            self.convs.append(conv)
            # self.pools.append(MaxPool1d(4, stride = 2))
            self.batch_norms.append(BatchNorm(80))
        
        self.lin = Sequential(Linear(40 * 80, 20 *40), 
                              MaxPool1d(2), ReLU(), Dropout(0.2),
                              Linear(20 * 20, 100), 
                              MaxPool1d(2), ReLU(), Dropout(0.2),
                              Linear(50, 20))

    def forward(self, x, edge_index, batch):        
        x = self.node_emb(x.squeeze()).view(-1, 80)
        
        for conv, batch_norm in zip(self.convs, self.batch_norms):            
            x = F.relu(batch_norm(conv(x, edge_index)))
        # x = self.mlp(x)       
        # return global_max_pool(x, batch)
        return self.lin(x.view(-1, 40 * 80))
    
class G2Dist_GCNConv_Global(torch.nn.Module):
    def __init__(self):
        super(G2Dist_GCNConv_Global, self).__init__()

        self.node_emb = Embedding(10000, 2)
        
        # self.pre = GCNConv(in_channels = 40, out_channels = 80)
        
        self.convs = ModuleList()
        self.pools = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(2):            
            conv = GCNConv(in_channels=80, out_channels=80, 
                           add_self_loops = True, bias = True)
            self.convs.append(conv)
            # self.pools.append(MaxPool1d(4, stride = 2))
            self.batch_norms.append(BatchNorm(80))
        
        # self.lin = Sequential(Linear(80, 40), 
        #                       ReLU(), Dropout(0.2),
        #                       Linear(40 , 20))
        self.lin = Sequential(Linear(80, 20))

    def forward(self, x, edge_index, batch):        
        x = self.node_emb(x.to(torch.long)).view(-1, 80)
        # x = self.pre(x.squeeze().to(torch.float), edge_index)
        
        for conv, batch_norm in zip(self.convs, self.batch_norms):            
            x = F.relu(batch_norm(conv(x, edge_index)))
        # x = self.mlp(x)       
        x = global_max_pool(x, batch)
        return self.lin(x)
    
class G2Dist_PNAConv(torch.nn.Module):
    def __init__(self, deg):
        super(G2Dist_PNAConv, self).__init__()

        self.node_emb = Embedding(10000, 2)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(2):
            conv = PNAConv(in_channels=80, out_channels=80,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           # edge_dim=50, 
                           towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(80))
            
        self.lin = Sequential(Linear(80, 20))

    def forward(self, x, edge_index, batch): #, edge_attr, batch):
        
        x = self.node_emb(x.squeeze()).view(-1, 80)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = batch_norm(conv(x, edge_index)).relu()
        x = global_max_pool(x, batch)
        return self.lin(x)