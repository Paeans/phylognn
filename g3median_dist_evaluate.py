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

import matplotlib.pyplot as plt

from gene_graph_dataset import G3MedianDataset

from dcj_comp import dcj_dist
from genome_file import mat2adj, dict2adj, pred_pair


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpuid", type=int, default = 0)
# parser.add_argument("--run", type=int)
parser.add_argument("--seqlen", type=int)
parser.add_argument("--rate", type=float, default = 0.9)
parser.add_argument("--samples", type=int, default = 1000)
parser.add_argument("--epoch", type=int, default = 100)

parser.add_argument("--shuffle", type=int, default = 1)
parser.add_argument("--vals", type=int, default = 200)
parser.add_argument("--valr", type=float, default = 0.9)
args = parser.parse_args()



dataset = G3MedianDataset('dataset_g3m_exps', args.seqlen, int(args.seqlen * args.rate), args.samples)
val_dataset = G3MedianDataset('val_seq_g3m', args.seqlen, int(args.seqlen * args.valr), args.vals)
val_seq = torch.load(f'val_seq_g3m_3_{args.seqlen}_{int(args.seqlen * args.valr)}_{args.vals}/raw/g3raw_{args.seqlen}_{int(args.seqlen * args.valr)}.pt')

train_batch, test_batch, val_batch = 64, 64, 8
device = torch.device('cuda:' + str(args.gpuid) if torch.cuda.is_available() else 'cpu')
in_channels, out_channels = 75, 128

dataset = dataset.shuffle()

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

model = G3Median_VGAE(G3Median_GCNConv(in_channels, out_channels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                              min_lr=0.00001,verbose=True)

# train_dataset = dataset[:int(len(dataset) * 1.0)]

train_loader = DataLoader(dataset, batch_size = train_batch, shuffle=True)

# start_time = time.time()


low_mean = 0
for seqs in val_seq[0]:
    low_mean += np.ceil(sum([dcj_dist(seqs[0], seqs[1])[-1], 
                            dcj_dist(seqs[0], seqs[2])[-1], 
                            dcj_dist(seqs[1], seqs[2])[-1]])/2)
low_mean = low_mean / len(val_seq[0])

writer = SummaryWriter(log_dir='dist_g3median_' f'{args.seqlen:0>4}' '/' f'{args.samples:0>4}' '_r' 
                       f'{args.rate:0>4.2f}' '_' f'{args.valr:0>4.2f}' '_' 'run_' f'{args.vals:0>4}')

final_dist = np.inf
final_list = []

print(f'{time.ctime()} -- seqlen:{args.seqlen:0>4} '
      f'rate:{args.rate:.2f} samples:{args.samples:0>5} -- events: {int(args.seqlen * args.valr):0>4}')

for epoch in range(1, args.epoch + 1):
    loss = train(model, train_loader)
    scheduler.step(loss)
        
    model.eval()
    result_list = []
    for vd, vs in zip(val_dataset, val_seq[0]):
        d = vd.to(device)
        z = model.encode(d.x, d.edge_index)
        res = model.decoder.forward_all(z).detach().cpu().numpy()
        
        seq_dist = pred_pair(res, vs, 20, rate = 0.7)
        dist_list = [x[2] for x in seq_dist]
        res_pair = seq_dist[np.argmin(dist_list)]
        result_list.append(res_pair)
        
    mean_dist = np.mean([x[2] for x in result_list])
    # print(mean_dist, epoch)
    writer.add_scalar('dist/mean', mean_dist - low_mean, epoch)
    
    if mean_dist < final_dist:
        final_dist = mean_dist
        final_list = result_list
        torch.save(model.state_dict(), 'model_params/' f'{args.seqlen:0>4}' f'_{int(args.seqlen * args.valr):0>4}.pth')
        
print(mean_dist - low_mean, final_list)
torch.save(final_list, f'g3median_final_list/' f'{args.seqlen:0>4}' '_' f'{args.valr}' 
           '_' f'{int(time.time()):0>10}.pt')
