import time
import torch

from phylognn_model import G2Braph_GCNConv
from gene_graph_dataset import G2BraphDataset

from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader

import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import roc_auc_score, average_precision_score

from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpuid", type=int)
parser.add_argument("--run", type=int)
parser.add_argument("--seqlen", type=int)
args = parser.parse_args()

gpuid = args.gpuid

train_p, test_p = 0.7, 0.2
train_batch = 25
test_batch, val_batch = 8, 8

dataset = G2BraphDataset('dataset_g2b', 10, 10, args.seqlen//10).shuffle()
data_size = len(dataset)
train_size, test_size = (int)(data_size * train_p), (int)(data_size * test_p)

print(data_size)

train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:(train_size + test_size)]
val_dataset = dataset[(train_size + test_size):]

train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch)
val_loader = DataLoader(val_dataset, batch_size=val_batch)

device = torch.device('cuda:' + str(gpuid) if torch.cuda.is_available() else 'cpu')

model = G2Braph_GCNConv().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                              min_lr=0.00001)

def train(train_dataset):
    model.train()
    
    total_loss = 0
    for data in train_dataset:
        data = data.to(device)
        optimizer.zero_grad()
        
        res = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy(res.squeeze(), data.node_label.to(torch.float))
        loss.backward()
        
        total_loss += loss
        optimizer.step()
        
    return total_loss / len(train_dataset)

@torch.no_grad()
def validate(test_dataset):
    model.eval()
    
    tloss = 0
    for data in test_dataset:
        data = data.to(device)
        res = model(data.x, data.edge_index)
        
        # y, pred = data.node_label.cpu().numpy(), res.squeeze().cpu().numpy()
        
        tloss += F.binary_cross_entropy(res.squeeze(), data.node_label.to(torch.float))
        
    return tloss / len(test_dataset)

@torch.no_grad()
def test(test_dataset):
    model.eval()
    
    auc, ap, counter = 0, 0, 0
    for data in test_dataset:
        data = data.to(device)
        res = model(data.x, data.edge_index)
        
        y, pred = data.node_label.cpu().numpy(), res.squeeze().cpu().numpy()
        if y.sum() == 0 or y.sum() == len(y):
            continue
        counter += 1
        auc += roc_auc_score(y, pred)
        ap += average_precision_score(y, pred)
        
    return auc/counter, ap/counter

writer = SummaryWriter(log_dir='runs_g2b_10/'+ str(args.seqlen) +'_gcn_run' + str(args.run))

for epoch in range(1, 201):
    train_loss = train(train_loader)
    val_loss = validate(val_loader)
    
    scheduler.step(val_loss)
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/validate', val_loss, epoch)
    
    auc, ap = test(test_dataset)
    writer.add_scalar('AUC/validate', auc, epoch)
    writer.add_scalar('AP/validate', ap, epoch)
    
    if epoch % 10 == 0:
        print(f'{time.ctime()}  '
              f'Epoch: {epoch:04d}, train Loss: {train_loss:.4f}, '
              f'val Loss: {val_loss:.4f}, auc: {auc:.4f}, '
              f'ap: {ap:.4f}')
        
writer.close()