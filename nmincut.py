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
from models import DGI, LogReg
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--temp', dest='temp', type=float, default=1,
        help='Temperature for gumbel sinkhorn, default=1')
parser.add_argument('--hard',action="store_true",
        help='Hard assignment of gumbel softmax') 
parser.add_argument('--fprocess',action="store_true",
        help='Process features') 
parser.add_argument('--epochs', dest='num_epochs', type=int, default=100000,
        help='Number of epochs to train, default=100000.')
parser.add_argument('--embed_size', type=int, default=128,
        help='Embedding size, default=128.')
parser.add_argument('--lr', dest='lr', type=float, default=0.001,
        help='Learning rate, default=0.001.')
parser.add_argument('--weight_decay', type=float, default=0,
        help='Weight decay, default=0.')
parser.add_argument('--dataset')

args =  parser.parse_args()


num_epochs = args.num_epochs
lr = args.lr
weight_decay = args.weight_decay

class GCNet(nn.Module):
    def __init__(self, num_features, num_embedding = 128):
        super(GCNet, self).__init__()
        self.conv = GCNConv(num_features, num_embedding, cached=True)
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x

adj, features, labels, idx_train, idx_val, idx_test = process.load_data(args.dataset)
if args.fprocess:
    features, _ = process.preprocess_features(features)
    features = np.array(features)
else:
    features = features.toarray()

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]
labels = np.argmax(labels, 1)
xent = nn.CrossEntropyLoss()
model = GCNet(ft_size, args.embed_size)
adj = Variable(torch.FloatTensor(adj.toarray()), requires_grad=False)
features = Variable(torch.FloatTensor(features), requires_grad=False)
labels = Variable(torch.LongTensor(labels), requires_grad=False)
edge_index = torch.transpose(adj.nonzero(),0,1)
edge_index = edge_index.long()
#adj += torch.eye(adj.shape[0])
if torch.cuda.is_available():
    model = model.cuda()
    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    edge_index = edge_index.cuda()

optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

smallest_loss = 1e20
embeddings_np = None 
best_at = 0

for epoch in tqdm(range(num_epochs)):
    model.train()
    model.zero_grad()
    embeddings = model(features, edge_index)
    assign_tensor = gumbel_softmax(embeddings, temp=args.temp, hard=args.hard)
    #assign_tensor =  F.softmax(embeddings,-1)
    assign_tensor_t = torch.transpose(assign_tensor, 0, 1)
    super_adj = assign_tensor_t @ adj @ assign_tensor # A' = S^T*A*S
    vol = super_adj.sum(1)
    diag = torch.diagonal(super_adj)
    norm_cut = (vol - diag)/(vol+1e-20)
    #print(torch.max(norm_cut), torch.min(norm_cut))
    loss = norm_cut.sum() + torch.sqrt(((norm_cut-norm_cut.mean())**2).sum()) *10

    loss.backward()
    optimizer.step()
    # if loss.item() < smallest_loss:
    if epoch %100 == 0:
        #import pdb;pdb.set_trace()
        # smallest_loss = loss.item()
        print(epoch, loss.item(), torch.max(norm_cut), torch.min(norm_cut))
        embeddings = embeddings.detach()
        X_train = embeddings[idx_train]
        Y_train = labels[idx_train]
        X_test = embeddings[idx_test]
        Y_test = labels[idx_test]
        X_val = embeddings[idx_val]
        Y_val = labels[idx_val]
        tot = torch.zeros(1)
        tot = tot.cuda()
        totv = torch.zeros(1)
        totv = totv.cuda()
        accs = []
        accsv = []
        for _ in range(50):
            log = LogReg(args.embed_size, nb_classes)
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
            log.cuda()

            pat_steps = 0
            best_acc = torch.zeros(1)
            best_acc = best_acc.cuda()
            for _ in range(100):
                log.train()
                opt.zero_grad()

                logits = log(X_train)
                loss = xent(logits, Y_train)
                
                loss.backward()
                opt.step()

            logits = log(X_test)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == Y_test).float() / Y_test.shape[0]
            accs.append(acc * 100)
            #print(acc)
            tot += acc

            logits = log(X_val)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == Y_val).float() / Y_val.shape[0]
            accsv.append(acc * 100)
            #print(acc)
            totv += acc

        #print('Average accuracy:', tot / 50)

        accs = torch.stack(accs)
        print(accs.mean().item(), accs.std().item())

        accsv = torch.stack(accsv)
        print(accsv.mean().item(), accsv.std().item())
# import pdb;pdb.set_trace()
