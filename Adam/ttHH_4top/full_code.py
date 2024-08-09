def adam_print(string):
    with open('output.txt','a') as file:
        file.write(string+'\n')

import numpy as np
adam_print('imported numpy')

import torch
from torch import nn
import torch.nn.functional as F
adam_print('imported torch')


from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
adam_print('imported torch_geometric')

import sklearn
from sklearn import preprocessing
from torcheval.metrics import BinaryAccuracy
adam_print('finished imports')

# p = 'C:/Users/HP/Documents/'
train_dataset = torch.load('data/train_dataset_equal.pth')
test_dataset = torch.load('data/test_dataset_equal.pth')

adam_print('loaded dataset')
adam_print(f'Number of training graphs: {len(train_dataset)}')
adam_print(f'Number of test graphs: {len(test_dataset)}')


train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
scaler_loader = [i for i in DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)][0]
scaler = preprocessing.StandardScaler()
scaler.fit(scaler_loader.x)

adam_print('loaders and scalers fitted')


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_convs):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()  # Use ModuleList to store layers
        self.convs.append(GCNConv(5, hidden_channels))
        for _ in range(num_convs - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, 1)  # Linear layer for final output
        # GraphConv
        # GCNConv


    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.01, training=self.training)
        x = self.lin(x)

        return torch.nn.functional.sigmoid(x)
    def __str__(self):
        return 'GCN'
    
class Graph(torch.nn.Module):
    def __init__(self, hidden_channels, num_convs):
        super(Graph, self).__init__()
        self.convs = nn.ModuleList()  # Use ModuleList to store layers
        self.convs.append(GraphConv(5, hidden_channels))
        for _ in range(num_convs - 1):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, 1)  # Linear layer for final output
        # GraphConv
        # GCNConv


    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.01, training=self.training)
        x = self.lin(x)

        return torch.nn.functional.sigmoid(x)
    def __str__(self):
        return 'Graph'
    #GATConv

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_convs):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()  # Use ModuleList to store layers
        self.convs.append(GATConv(5, hidden_channels))
        for _ in range(num_convs - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, 1)  # Linear layer for final output
        # GraphConv
        # GCNConv


    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.01, training=self.training)
        x = self.lin(x)

        return torch.nn.functional.sigmoid(x)
    def __str__(self):
        return 'GAT'
    
class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels, num_convs):
        super(SAGE, self).__init__()
        self.convs = nn.ModuleList()  # Use ModuleList to store layers
        self.convs.append(SAGEConv(5, hidden_channels))
        for _ in range(num_convs - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, 1)  # Linear layer for final output
        # GraphConv
        # GCNConv



    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.01, training=self.training)
        x = self.lin(x)

        return torch.nn.functional.sigmoid(x)
    def __str__(self):
        return 'SAGE'
    
adam_print('GNN classes defined')



def train_test(model, optimizer, criterion, loader, Type = '', output=False):
    if Type == 'train':
        model.train()
    else:
        model.eval()
    total_loss = 0
    metric = BinaryAccuracy()
    #out_numpy = np.array([])
    for data in loader:  # Iterate in batches over the training dataset. 
        x_transformed = torch.tensor(scaler.transform(data.x),dtype=torch.float32)
        out = model(x_transformed, data.edge_index, data.batch)  # Perform a single forward pass.
        metric.update(out.flatten(), data.y)
        # loss = torch.sum(criterion(out, data.y.reshape(-1,1))*data.w.reshape(-1,1)) / data.w.sum()# Compute the loss.
        loss = criterion(out, data.y.reshape(-1,1))
        if Type == 'train':
            optimizer.zero_grad()  # Clear gradients.   
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

        total_loss += loss.item() * data.size(0)
    total_loss /= len(loader.dataset)
    accuracy = metric.compute().item()
    if output:
        adam_print(f'{output} accuracy: {100*accuracy:.3f}% , loss: {total_loss:.6f}')
    return model, accuracy, total_loss

adam_print('test_train defined')


def each_run(class_type, criterion, layers, hidden_dims, learn_rate, num_epochs):
    model = class_type(hidden_channels=hidden_dims, num_convs=layers)
    title = f'{str(model)}_{layers}_{hidden_dims}_{learn_rate}'
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    test_results =  np.empty((0,2))
    train_results =  np.empty((0,2))
    count = 0
    best_acc = 0
    best_loss = 0
    for epoch in range(num_epochs[1]):
        count+=1
        adam_print(f'Epoch: {epoch:03d}')
        model, acc, loss = train_test(model, optimizer, criterion, train_loader, 'train')
        train_results = np.vstack((train_results, [acc,loss]))
        model, acc, loss = train_test(model, optimizer, criterion, test_loader, 'test')
        test_results = np.vstack((test_results, [acc,loss]))
        if acc>best_acc:
            adam_print(f'saving model: {acc} {loss}')
            torch.save(model.state_dict(), f'models/model_equal_{title}.pth')
            count = 0
            best_acc = acc
        if best_loss == 0 or loss<best_loss:
            best_loss = loss
            count = 0
        if epoch>num_epochs[0] and count > 10:
            break
    
    with open('train_stats', 'a') as file:
        string = f'{title}: {train_results.tolist()}: {test_results.tolist()}\n'
        file.write(string)


adam_print('each run defined')
criterion = torch.nn.BCELoss(reduction = 'mean')
def do_it_all(class_list, layers_list, hidden_dims_list, num_epochs, learn_rate):
    for class_type in class_list:
        for layers in layers_list:
            for hidden_dims in hidden_dims_list:
                adam_print(f'{class_type.__str__(class_type)} {layers} {hidden_dims} {learn_rate} ')
                each_run(class_type, criterion, layers, hidden_dims, learn_rate, num_epochs)

def do_it_all_specific(schedule, num_epochs):
    for process in schedule:
        adam_print(f'{process[0].__str__(process[0])} {process[1]} {process[2]} {process[3]} ')
        each_run(process[0], criterion, process[1], process[2], process[3], num_epochs)
adam_print('all functions defined')

schedule = [[Graph, 2, 64, 0.001], [Graph, 2, 32, 0.001], [Graph, 2, 128, 0.001]]


# class_list = [Graph, GAT, SAGE, GCN]
# layers_list = [2,3,4,5]
# hidden_dims_list = [16,32,64,128,256]

# class_list = [SAGE]
# layers_list = [3]
# hidden_dims_list = [64]


do_it_all_specific(schedule, (20,80))
adam_print('program finished')

