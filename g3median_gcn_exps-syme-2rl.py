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

from genome_graph import gen_g2g_graph
from gene_graph_dataset import G3MedianDataset
from phylognn_model import G3Median_GCNConv, G3Median_VGAE

from sklearn.metrics import (roc_auc_score, roc_curve,
                             average_precision_score, 
                             precision_recall_curve,
                             f1_score, matthews_corrcoef)

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from dcj_comp import dcj_dist
from genome_file import mat2adj

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpuid", type=int, default = 0)
# parser.add_argument("--run", type=int)
parser.add_argument("--seqlen", type=int)
parser.add_argument("--rate", type=float, default = 0.1)
parser.add_argument("--samples", type=int, default = 1000)
parser.add_argument("--epoch", type=int, default = 1000)
parser.add_argument("--cvsplit", type=int, default = 5)
parser.add_argument("--freq", type=int, default = 20)
parser.add_argument("--shuffle", type=int, default = 1)
parser.add_argument("--vals", type=int, default = 100)
parser.add_argument("--valr", type=float, default = 0.1)
args = parser.parse_args()


gpuid = args.gpuid # 0

# train_p, test_p, val_p = 0.7, 0.2, 0.1
train_batch, test_batch, val_batch = 256, 64, 8

device = torch.device('cuda:' + str(gpuid) if torch.cuda.is_available() else 'cpu')

dataset = G3MedianDataset('dataset_g3m', args.seqlen, int(args.seqlen * args.rate), args.samples)
# val_dataset = G3MedianDataset('val_seq_g3m', args.seqlen, int(args.seqlen * args.valr), args.vals)
val_seq, tar_seq = torch.load(f'val_seq_g3m_3_{args.seqlen}_{int(args.seqlen * args.valr)}_{args.vals}/'
                              f'raw/g3raw_{args.seqlen}_{int(args.seqlen * args.valr)}.pt')
val_dataset = [gen_g2g_graph(s, t) for s,t in zip(val_seq, tar_seq)]

in_channels, out_channels = None, 128

dataset = dataset.shuffle()

# from torch_geometric.data import Batch
def train(model, train_loader):
    model.train()
    
    total_loss = 0
    for data in train_loader:    
        optimizer.zero_grad()
        data = data.to(device)
        
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss_wt(z, data.pos_edge_label_index, data.neg_edge_label_index, 2, 1) * 5
        loss = loss + (1 / data.num_nodes) * model.kl_loss() * 0.5
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
def median_score(model, val_dataset, val_sequence):
    model.eval()
    count, num = 0, 0
    for d, seqs in zip(val_dataset, val_sequence):
        d = d.to(device)
        z = model.encode(d.x, d.edge_index)
        res = model.decoder.forward_all(z).detach().cpu().numpy()
        pred_seqs = mat2adj(res)
        
        pred_dist = min([sum([dcj_dist(pred, s)[-1] for s in seqs]) for pred in pred_seqs])
        low_dist = np.ceil(sum([dcj_dist(seqs[0], seqs[1])[-1], 
                                dcj_dist(seqs[0], seqs[2])[-1], 
                                dcj_dist(seqs[1], seqs[2])[-1]])/2)

        # print(f'{pred_dist:>3} -- {low_dist:<3}')
        diff = pred_dist - low_dist
        if diff == 0:
            num += 1
        count += diff

    return count / len(val_dataset), num / len(val_dataset)

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
        loss += model.recon_loss_wt(z, data.pos_edge_label_index, data.neg_edge_label_index, 2, 1)
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

print(f'{time.ctime()} -- seqlen:{args.seqlen:0>4} '
      f'rate:{args.rate:.2f} samples:{args.samples:0>5} -- fold: {args.vals:0>4}')

model = G3Median_VGAE(G3Median_GCNConv(in_channels, out_channels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                              min_lr=0.00001,verbose=True)

writer = SummaryWriter(log_dir='exps_g3median_' f'{args.seqlen:0>4}' '/e' f'{args.samples:0>5}' '_r' 
                       f'{args.rate:0>3.1f}' '_' 'run_' f'{args.vals:0>4}')

train_loader = DataLoader(dataset, batch_size = train_batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = val_batch)

start_time = time.time()
for epoch in range(1, args.epoch + 1):

    loss = train(model, train_loader)
    tloss = val(model, val_loader)
    scheduler.step(tloss)

    writer.add_scalar('loss/train', loss, epoch)
    writer.add_scalar('loss/val', tloss, epoch)

    # if epoch % args.freq != 0:
    #     continue

    score, acc = median_score(model, val_dataset, val_seq)

    y_list, pred_list = predict(model, val_loader)
    auc, ap = auc_ap(y_list, pred_list)

    writer.add_scalar('auc/test', auc, epoch)
    writer.add_scalar('ap/test', ap, epoch)
    writer.add_scalar('score/test', score, epoch)
    writer.add_scalar('acc/test', acc, epoch)

end_time = time.time()
print(f'{time.ctime()} -- seqlen:{args.seqlen:0>4} '
      f'rate:{args.rate:.2f} samples:{args.samples:0>5} -- fold: {args.vals:0>2}'
     f' -- {(end_time - start_time)/args.epoch:>10.3f}s * {args.epoch:0>4} epoches')
writer.close()
    
# torch.save(y_pred_res, 
#            f'y_pred/ldel' f'{args.seqlen:0>4}' 
#            '-r' f'{args.rate:0>3.1f}' 
#            '-s' f'{args.samples:0>5}' 
#            '-' f'{int(time.time()):0>10}.pt')
