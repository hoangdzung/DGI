import torch 
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

from torch_geometric.nn import GCNConv
from sklearn.linear_model import LogisticRegression

import numpy as np
from tqdm import tqdm

from gumbel import gumbel_softmax
from utils import process

num_epochs = 100000
lr = 0.001
weight_decay = 0

class GCNet(nn.Module):
    def __init__(self, num_features, num_embedding = 128):
        super(GCNet, self).__init__()
        self.conv = GCNConv(num_features, num_embedding, cached=True)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x

adj, features, labels, idx_train, idx_val, idx_test = process.load_data('cora')
#features, _ = process.preprocess_features(features)

features = features.toarray()
#features=np.array(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]
labels = np.argmax(labels, 1)

model = GCNet(ft_size)
adj = Variable(torch.FloatTensor(adj.toarray()), requires_grad=False)
features = Variable(torch.FloatTensor(features), requires_grad=False)
edge_index = torch.transpose(adj.nonzero(),0,1)
edge_index = edge_index.long()

if torch.cuda.is_available():
    model = model.cuda()
    adj = adj.cuda()
    features = features.cuda()
    edge_index = edge_index.cuda()

optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

smallest_loss = 1e20
embeddings_np = None 
best_at = 0

for epoch in tqdm(range(num_epochs)):
    model.train()
    model.zero_grad()
    embeddings = model(features, edge_index)
    assign_tensor = gumbel_softmax(embeddings, temp=0.1,hard=True)
    assign_tensor_t = torch.transpose(assign_tensor, 0, 1)
    super_adj = assign_tensor_t @ adj @ assign_tensor # A' = S^T*A*S
    vol = super_adj.sum(1)
    diag = torch.diagonal(super_adj)
    norm_cut = (vol - diag)/(vol+1e-20)
    loss = norm_cut.sum() 

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()
    if loss.item() < smallest_loss:
        smallest_loss = loss.item()
    embeddings_np = embeddings.cpu().detach().numpy()
    X_train = embeddings_np[idx_train]
    Y_train = labels[idx_train]
    X_test = embeddings_np[idx_test]
    Y_test = labels[idx_test]
    clf = LogisticRegression(solver="lbfgs", max_iter=4000)
    clf.fit(X_train, Y_train)
    print(loss.item(), clf.score(X_test, Y_test))


# import pdb;pdb.set_trace()a
