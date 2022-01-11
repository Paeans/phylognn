import time

import numpy as np

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from torch_geometric.nn import VGAE
from torch_geometric.loader import DataLoader
from torch_geometric.utils import (degree, negative_sampling, 
                                   batched_negative_sampling,
                                  add_self_loops, to_undirected)

from torch.utils.tensorboard import SummaryWriter

from gene_graph_dataset import G3MedianDataset
from phylognn_model import G3Median_GCNConv, G3Median_VGAE

from sklearn.metrics import (roc_auc_score, roc_curve,
                             average_precision_score, 
                             precision_recall_curve,
                             f1_score, matthews_corrcoef)

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpuid", type=int, default = 0)
# parser.add_argument("--run", type=int)
parser.add_argument("--seqlen", type=int)
parser.add_argument("--rate", type=float, default = 0.1)
parser.add_argument("--samples", type=int, default = 1000)
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--cvsplit", type=int, default=5)
args = parser.parse_args()


gpuid = args.gpuid # 0

# train_p, test_p, val_p = 0.7, 0.2, 0.1
train_batch, test_batch, val_batch = 128, 64, 8

device = torch.device('cuda:' + str(gpuid) if torch.cuda.is_available() else 'cpu')

dataset = G3MedianDataset('dataset_g3m', args.seqlen, int(args.seqlen * args.rate), args.samples)

in_channels, out_channels = None, 16

# data_size = len(dataset)
# train_size, test_size, val_size = ((int)(data_size * train_p), 
#                                    (int)(data_size * test_p), 
#                                    (int)(data_size * val_p))

# print(f'dataset size: {data_size:0>5}')
dataset = dataset.shuffle()

# train_dataset = dataset[:train_size]
# test_dataset = dataset[train_size:(train_size + test_size)]
# val_dataset = dataset[(train_size + test_size):(train_size + test_size + val_size)]

# test_dataset = list(test_dataset)
# for t in test_dataset:
#     t.pos_edge_label_index = add_self_loops(to_undirected(t.pos_edge_label_index))[0]
#     t.neg_edge_label_index = negative_sampling(t.pos_edge_label_index, 
#                                         t.num_nodes,
#                                         t.num_nodes**2)
# train_dataset = list(train_dataset)
# for t in train_dataset:
#     t.pos_edge_label_index = add_self_loops(to_undirected(t.pos_edge_label_index))[0]
#     t.neg_edge_label_index = negative_sampling(t.pos_edge_label_index, 
#                                         t.num_nodes,
#                                         t.num_nodes**2)
# val_dataset = list(val_dataset)
# for t in val_dataset:
#     t.pos_edge_label_index = add_self_loops(to_undirected(t.pos_edge_label_index))[0]
#     t.neg_edge_label_index = negative_sampling(t.pos_edge_label_index, 
#                                         t.num_nodes,
#                                         t.num_nodes**2)
    

# from torch_geometric.data import Batch
def train(model, train_loader):
    model.train()
    
    total_loss = 0
    for data in train_loader:    
        optimizer.zero_grad()
        data = data.to(device)
        
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.pos_edge_label_index, data.neg_edge_label_index) * 2
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        
        total_loss += loss
    return total_loss/len(train_loader)

# @torch.no_grad()
# def test(model, test_loader):
#     model.eval()
#     auc, ap = 0, 0
    
#     for data in test_loader:
        
#         data = data.to(device)
        
#         z = model.encode(data.x, data.edge_index)
#         # loss += model.recon_loss(z, data.pos_edge_label_index, data.neg_edge_label_index)
#         tauc, tap = model.test(z, data.pos_edge_label_index) #, data.neg_edge_label_index)
        
#         auc += tauc
#         ap += tap
        
#     return auc/len(test_loader), ap/len(test_loader)

@torch.no_grad()
def predict(model, test_loader):
    model.eval()
    y_list, pred_list = [], []
        
    for data in test_loader:
        
        data = data.to(device)
        
        z = model.encode(data.x, data.edge_index)
        # loss += model.recon_loss(z, data.pos_edge_label_index, data.neg_edge_label_index)
        y, pred = model.pred(z, data.pos_edge_label_index, data.neg_edge_label_index)
        
        y_list.append(y)
        pred_list.append(pred)
        
    return y_list, pred_list

@torch.no_grad()
def val(model, val_loader):
    model.eval()
    loss = 0
    
    for data in val_loader:        
        data = data.to(device)        
        z = model.encode(data.x, data.edge_index)        
        loss += model.recon_loss(z, data.pos_edge_label_index, data.neg_edge_label_index)
        # tauc, tap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
                
    return loss/len(val_loader)

def auc_ap(y_list, pred_list):
    pred_accuracy = [[roc_auc_score(y, pred), average_precision_score(y, pred)]
                     for y, pred in zip(y_list, pred_list)]
    auc, ap = np.mean(pred_accuracy, axis = 0)
    return auc, ap

def cal_accuracy(y_list, pred_list):
    # pred_accuracy = np.zeros((len(y_list), 2))
    # for i in range(len(y_list)):
    #     y, pred = y_list[i], pred_list[i]
    #     pred_accuracy[i] = [roc_auc_score(y, pred), 
    #                         average_precision_score(y, pred)]
    
    figsize = (6,6)
        
    y, pred = np.concatenate([[t, p] for t, p in zip(y_list, pred_list)], axis = -1)
    auc, ap = roc_auc_score(y, pred), average_precision_score(y, pred)
        
    auc_figure = plt.figure(figsize=figsize)
    
    fpr, tpr, _ = roc_curve(y, pred)
    plt.plot(fpr, tpr, color='g', lw=0.3)
    # for i in range(len(y_list)):
    #     y, pred = y_list[i], pred_list[i]
    #     fpr, tpr, _ = roc_curve(y, pred)
    #     plt.plot(fpr, tpr, color='g', lw=0.3)
    
    plt.plot([0, 1], [0, 1], color="navy", lw=0.3, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'Receiver Operating Characteristic ({auc:.4f})')
    # plt.legend(loc="lower right")
    
    ap_figure = plt.figure(figsize=figsize)
    
    prc, rec, _ = precision_recall_curve(y, pred)
    plt.plot(rec, prc, color='c', lw=0.3)
    # for i in range(len(y_list)):
    #     y, pred = y_list[i], pred_list[i]
    #     prc, rec, _ = precision_recall_curve(y, pred)
    #     plt.plot(rec, prc, color='c', lw=0.3)
        
    plt.plot([0, 1], [0, 1], color="navy", lw=0.3, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f'Precision-Recall Curve ({ap:.4f})')
    
    
    return [auc, ap], [auc_figure, ap_figure] #, ('auc', 'ap')

y_pred_res = []
counter = 1
for train_index, test_index in KFold(n_splits = args.cvsplit).split(dataset):
    
    print(f'{time.ctime()} -- seqlen:{args.seqlen:0>4} '
          f'rate:{args.rate:.2f} samples:{args.samples:0>5} -- fold: {counter:0>2}')
    
    model = G3Median_VGAE(G3Median_GCNConv(in_channels, out_channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15,
                                  min_lr=0.00001)

    writer = SummaryWriter(log_dir='runs_g3median_' f'{args.seqlen:0>4}' '/s' f'{args.samples:0>5}' '_r' 
                           f'{args.rate:0>3.1f}' '_' 'run' f'{counter:0>2}')
    
    train_dataset = dataset[train_index]
    test_dataset = dataset[test_index]
    
    train_dataset = train_dataset[:int(len(train_dataset) * 0.9)]
    val_dataset = train_dataset[int(len(train_dataset) * 0.9):]
    
    train_loader = DataLoader(train_dataset, batch_size = train_batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = test_batch)
    val_loader = DataLoader(val_dataset, batch_size = val_batch)
    
    start_time = time.time()
    
    y_pred = None
    p_auc, p_ap = 0, 0
    for epoch in range(1, args.epoch + 1):
        
        loss = train(model, train_loader)
        tloss = val(model, val_loader)
        scheduler.step(tloss)

        writer.add_scalar('loss/train', loss, epoch)
        writer.add_scalar('loss/val', tloss, epoch)


        y_list, pred_list = predict(model, test_loader)
        # pred_acc, figures = cal_accuracy(y_list, pred_list)        
        # auc, ap = pred_acc
        
        # y_list, pred_list = predict(model, test_dataset)
        auc, ap = auc_ap(y_list, pred_list)
        
        writer.add_scalar('auc/test', auc, epoch)
        writer.add_scalar('ap/test', ap, epoch)
        
        # writer.add_figure('roc/test', figures[0], epoch)
        # writer.add_figure('pr/test', figures[1], epoch)
        
        if auc >= p_auc and ap >= p_ap:
            y_pred = np.concatenate([np.array([y, pred])
                                     for y, pred in zip(y_list, pred_list)], 
                                    axis = 1)
            p_auc, p_ap = auc, ap
        
    end_time = time.time()
    print(f'{time.ctime()} -- seqlen:{args.seqlen:0>4} '
          f'rate:{args.rate:.2f} samples:{args.samples:0>5} -- fold: {counter:0>2}'
         f' -- {(end_time - start_time)/args.epoch:>10.3f}s * {args.epoch:0>4} epoches')
    y_pred_res.append(y_pred)
    
    writer.close()
    counter += 1
    
torch.save(y_pred_res, 
           f'y_pred/lnew' f'{args.seqlen:0>4}' 
           '-r' f'{args.rate:0>3.1f}' 
           '-s' f'{args.samples:0>5}' 
           '-' f'{int(time.time()):0>10}.pt')
