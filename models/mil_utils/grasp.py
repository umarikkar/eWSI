import torch.nn as nn
from torch_geometric.nn import GCNConv
from einops import rearrange
import torch
import torch.nn.functional as F


class GRASP(nn.Module):

    def __init__(
        self, in_features, out_features, hidden_dim = 128, gcn_hidden_dim=256, num_subimgs=128, num_subimgs_val=-1, num_mags=3
    ):
        super().__init__()

        self.num_subimgs = num_subimgs
        self.num_subimgs_val = num_subimgs_val
            
        self.gcn1 = GCNConv(in_features, gcn_hidden_dim)
        self.gcn2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.gcn3 = GCNConv(gcn_hidden_dim, hidden_dim)

        self.edges_train = nn.Parameter(self.create_adj_mat(num_subimgs, num_mags), 
                                  requires_grad=False)
        
        if num_subimgs_val>0:
            self.edges_val = nn.Parameter(self.create_adj_mat(num_subimgs, num_mags), 
                                    requires_grad=False)

        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_features))

    def create_adj_mat(self, n, m):

        idxs_1 = torch.cat([torch.tril_indices(n,n,-1), 
                    torch.triu_indices(n,n,1)], dim=1)

        # for all 'm', all 'n' per each 'm' are connected ->
        idxs_n1 = torch.cat([idxs_1 + (n*i) for i in range(m)], dim=1)


        # for each n, all 'm' are connected ->
        idxs_2 = torch.cat([torch.tril_indices(m,m,-1), 
                            torch.triu_indices(m,m, 0)], dim=1)

        # for each n, all 'm' are connected ->
        idxs_n2 = torch.cat([idxs_2*n +i for i in range(n)], dim=1)

        edges = torch.cat([idxs_n1, idxs_n2], dim=1)

        return edges
    

    def forward(self, features):

        # reshape the features
        b, n, m, d = features.shape

        if self.training and self.num_subimgs>0:
            edges = self.edges_train
        elif self.num_subimgs_val>0:
            edges = self.edges_val
        else:
            edges = self.create_adj_mat(n,m).cuda()

        # you HAVE to rearrange in this setting to get it to work
        x = rearrange(features, 'b n m d -> b (m n) d')

        x = self.gcn1(x, edges)
        x = F.dropout(F.relu(x), training=self.training)

        x = self.gcn2(x, edges)
        x = F.dropout(F.relu(x), training=self.training)

        x = self.gcn3(x, edges)
        x = F.dropout(F.relu(x), training=self.training)

        x = rearrange(x, 'b (m n) d -> b n m d', m=m, n=n)

        x = x.mean(-2)
        x = x.mean(-2)

        logits = self.mlp(x)

        return logits
    