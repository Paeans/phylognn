import time

import torch

import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, PNAConv, BatchNorm, global_add_pool

from phylognn_model import G2Dist_GCNConv_Global, G2Dist_GCNConv_Small

from gene_graph_dataset import GeneGraphDataset

from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpuid", type=int)
parser.add_argument("--run", type=int)
parser.add_argument("--seqlen", type=int)
parser.add_argument("--weight", type=float)
parser.add_argument("--epoch", type=int)
parser.add_argument("--train", type=float)
parser.add_argument("--test", type=float)

args = parser.parse_args()

train_p, test_p = args.train, args.test
step = 5

dataset = GeneGraphDataset('dataset_adj', 20, step, graph_num = args.seqlen//step)
data_size = len(dataset)
train_size, test_size = (int)(data_size * train_p), (int)(data_size * test_p)

dataset = dataset.shuffle()
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:(train_size + test_size)]
val_dataset = dataset[(train_size + test_size):]

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512)
val_loader = DataLoader(val_dataset, batch_size=512)

device = torch.device('cuda:' + str(args.gpuid) if torch.cuda.is_available() else 'cpu')

model = G2Dist_GCNConv_Small().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = args.weight)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                              min_lr=0.00001)

writer = SummaryWriter(log_dir='runs_g2d_10/g2dist_adjone_20_05_' + f'{args.seqlen:0>5}' + '-small-run' + f'{args.run:0>2}')

loss_fn = CrossEntropyLoss()

def train(train_loader):
    model.train()

    total_loss, counter = 0, 0
    size = len(train_loader)
    for batch, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        #loss = (out.squeeze() - data.y).abs().sum()
        pred, y = out.softmax(axis = 1).argmax(axis = 1), data.y
        counter += (pred == y).sum().item()
        
        loss = loss_fn(out, data.y)
        
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        
    return total_loss / len(train_loader), counter

@torch.no_grad()
def test(loader):
    model.eval()

    total_error, counter = 0, 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        
        pred, y = out.softmax(axis = 1).argmax(axis = 1), data.y
        counter += (pred == y).sum().item()
        
        # total_error += (out.squeeze() - data.y).abs().sum().item()
        
        total_error += loss_fn(out, data.y).item()
        
    return total_error / len(loader), counter

import numpy as np
for epoch in range(1, args.epoch + 1):
    loss, train_counter = train(train_loader)
    test_mae, test_counter = test(test_loader)
    val_mae, _ = test(val_loader)
    
    scheduler.step(loss)
    
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Loss/test', test_mae, epoch)
    writer.add_scalar('Loss/val', val_mae, epoch)
    writer.add_scalar('Counter/train', train_counter/len(train_loader.dataset), epoch)
    writer.add_scalar('Counter/test', test_counter/len(test_loader.dataset), epoch)
    
    print(f'{time.ctime()}\t'
          f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
          f'Test: {test_mae:.4f}')

writer.close()