import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn import VGAE
from torch_geometric.loader import DataLoader
from torch_geometric.utils import (degree, negative_sampling, 
                                  add_self_loops, to_undirected)

from torch.utils.tensorboard import SummaryWriter

from gene_graph_dataset import G3MedianDataset
from phylognn_model import G3Median_GCNConv


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpuid", type=int)
parser.add_argument("--run", type=int)
parser.add_argument("--seqlen", type=int)
args = parser.parse_args()


gpuid = args.gpuid # 0

train_p, test_p, val_p = 0.7, 0.2, 0.1
train_batch, test_batch, val_batch = 128, 64, 8

device = torch.device('cuda:' + str(gpuid) if torch.cuda.is_available() else 'cpu')

dataset = G3MedianDataset('dataset_g3m', 10, 10, args.seqlen//10)

data_size = len(dataset)
train_size, test_size, val_size = ((int)(data_size * train_p), 
                                   (int)(data_size * test_p), 
                                   (int)(data_size * val_p))
print(data_size)
dataset = dataset.shuffle()
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:(train_size + test_size)]
val_dataset = dataset[(train_size + test_size):(train_size + test_size + val_size)]

test_dataset = list(test_dataset)
for t in test_dataset:
    t.pos_edge_label_index = add_self_loops(to_undirected(t.pos_edge_label_index))[0]
    t.neg_edge_label_index = negative_sampling(t.pos_edge_label_index, 
                                        t.num_nodes,
                                        t.num_nodes**2)
train_dataset = list(train_dataset)
for t in train_dataset:
    t.pos_edge_label_index = add_self_loops(to_undirected(t.pos_edge_label_index))[0]
    t.neg_edge_label_index = negative_sampling(t.pos_edge_label_index, 
                                        t.num_nodes,
                                        t.num_nodes**2)
val_dataset = list(val_dataset)
for t in val_dataset:
    t.pos_edge_label_index = add_self_loops(to_undirected(t.pos_edge_label_index))[0]
    t.neg_edge_label_index = negative_sampling(t.pos_edge_label_index, 
                                        t.num_nodes,
                                        t.num_nodes**2)
    
train_loader = DataLoader(train_dataset, batch_size = train_batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = test_batch)
val_loader = DataLoader(val_dataset, batch_size= val_batch)

in_channels, out_channels = dataset.num_features, 16

model = VGAE(G3Median_GCNConv(in_channels, out_channels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30,
                              min_lr=0.00001)

writer = SummaryWriter(log_dir='runs_g3m_10/g3median_' + str(args.seqlen) + '_gcn_aneg_syme_2rl_run' + str(args.run))

from torch_geometric.data import Batch
def train(train_loader):
    model.train()
    
    total_loss = 0
    for data in train_loader:
        
        data = data.to(device)
        optimizer.zero_grad()
        
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.pos_edge_label_index, data.neg_edge_label_index) * 2
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        
        total_loss += loss
    return total_loss/len(train_loader)

@torch.no_grad()
def test(test_loader):
    model.eval()
    auc, ap = 0, 0
    
    for data in test_loader:
        
        data = data.to(device)
        
        z = model.encode(data.x, data.edge_index)
        # loss += model.recon_loss(z, data.pos_edge_label_index, data.neg_edge_label_index)
        tauc, tap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
        
        auc += tauc
        ap += tap
        
    return auc/len(test_loader), ap/len(test_loader)

@torch.no_grad()
def val(val_loader):
    model.eval()
    loss = 0
    
    for data in val_loader:
        
        data = data.to(device)
        
        z = model.encode(data.x, data.edge_index)
        loss += model.recon_loss(z, data.pos_edge_label_index, data.neg_edge_label_index)
        # tauc, tap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
                
    return loss/len(val_loader)

for epoch in range(1, 1000 + 1):
    # print(f'{time.ctime()} - Epoch: {epoch:04d}')
    loss = train(train_loader)
    # print(f'{time.ctime()} - \t train loss: {loss:.6f}')
    tloss = val(val_loader)
    # print(f'{time.ctime()} - \t val   loss: {tloss:.6f}')
    if epoch > 800:
        scheduler.step(tloss)
    
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Loss/val', tloss, epoch)
    
    
    auc, ap = test(test_dataset)
    
    writer.add_scalar('AUC/test', auc, epoch)
    writer.add_scalar('AP/test', ap, epoch)
    
    if epoch % 50 == 0:
        print(f'{time.ctime()} - Epoch: {epoch:04d}        auc: {auc:.6f}, ap: {ap:.6f}')
        
writer.close()