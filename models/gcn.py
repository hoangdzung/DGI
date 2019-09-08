import torch 
import torch.nn.functional as F
from torch import nn
from layers import GCN, AvgReadout

class GCNet(torch.nn.Module):
    def __init__(self, n_in, n_h, activation):
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

    def forward(self, seq1, adj, sparse):
        h = self.gcn(seq1, adj, sparse)
        return h

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()
