import torch 
import torch.nn.functional as F
from gumbel import gumbel_softmax

def ncut_loss(adj, embeddings, temp=1.0, hard=True):
    # assign_tensor = F.gumbel_softmax(embeddings, tau=temp, hard=hard)
    assign_tensor = gumbel_softmax(embeddings, temp, hard)
    assign_tensor_t = torch.transpose(assign_tensor, 0, 1)
    super_adj = assign_tensor_t @ adj @ assign_tensor # A' = S^T*A*S
    vol = super_adj.sum(1)
    diag = torch.diagonal(super_adj)
    norm_cut = (vol - diag)/(vol+1e-20)
    loss = norm_cut.sum()
    return loss
    
