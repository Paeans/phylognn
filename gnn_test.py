import time

import torch

import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, PNAConv, BatchNorm, global_add_pool

from gene_graph_dataset import GeneGraphDataset

train_p, test_p = 0.7, 0.2

dataset = GeneGraphDataset('dataset', 100, 5)
data_size = len(dataset)
train_size, test_size = (int)(data_size * train_p), (int)(data_size * test_p)

dataset = dataset.shuffle()
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:(train_size + test_size)]
val_dataset = dataset[(train_size + test_size):]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)
val_loader = DataLoader(val_dataset, batch_size=128)

deg = torch.zeros(5, dtype=torch.long)
for data in train_dataset:
    d = degree(data.edge_index[1].type(torch.int64), 
               num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())
    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.node_emb = Embedding(21, 75)
        self.edge_emb = Embedding(4, 25)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=50, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            # conv = GCNConv(in_channels=75, out_channels=75)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))
            
        self.pre_lin = Linear(150,75)

        self.mlp = Sequential(Linear(75, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        
        x = torch.reshape(self.node_emb(x.squeeze()), (-1, 150))
        x = self.pre_lin(x)
        
        edge_attr = torch.reshape(self.edge_emb(edge_attr), (-1,50))

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            # x = F.relu(batch_norm(conv(x, edge_index)))
        x = global_add_pool(x, batch)        
        return self.mlp(x)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                              min_lr=0.00001)

def train(train_loader):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = (out.squeeze() - data.y).abs().mean()
        
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(loader):
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)

for epoch in range(1, 301):
    loss = train(train_loader)
    test_mae = test(test_loader)
    val_mae = test(val_loader)
    
    scheduler.step(val_mae)
    # if epoch % 10 == 0:
    print(f'{time.ctime()}\t'
          f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
          f'Test: {test_mae:.4f}')