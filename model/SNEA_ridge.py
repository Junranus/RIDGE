#!/usr/bin/env python
# encoding: utf-8

import torch
import scipy.sparse
import torch.nn.functional as F
from torch_geometric.utils import coalesce
from torch_geometric.utils import negative_sampling
from torch_geometric_signed_directed.nn.signed import SNEAConv
from torch_geometric.utils import structured_negative_sampling


class SNEA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, node_num, lamb=4, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.node_num = node_num
        self.lamb = lamb
        self.SAMPLING_RATIO = 1.0  # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                conv = SNEAConv(in_channels, hidden_channels//2, first_aggr=True)
            else:
                conv = SNEAConv(hidden_channels//2, hidden_channels//2, first_aggr=False)
            self.convs.append(conv)

        self.lin = torch.nn.Linear(2*hidden_channels, 3)
        self.feat_mask = self.construct_feat_mask(in_channels, init_strategy="constant")
        self.weight = torch.nn.Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()
        self.feat_mask = self.construct_feat_mask(self.in_channels, init_strategy="constant")
        self.weight.reset_parameters()

    def create_spectral_features(self, pos_edge_index, neg_edge_index, num_nodes):
        r"""Creates :obj:`in_channels` spectral node features based on
        positive and negative edges.

        Args:
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
            num_nodes (int, optional): The number of nodes, *i.e.*
                :obj:`max_val + 1` of :attr:`pos_edge_index` and
                :attr:`neg_edge_index`. (default: :obj:`None`)
        """
        from sklearn.decomposition import TruncatedSVD

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        N = edge_index.max().item() + 1 if num_nodes is None else num_nodes
        edge_index = edge_index.to(torch.device('cpu'))

        pos_val = torch.full((pos_edge_index.size(1), ), 2, dtype=torch.float)
        neg_val = torch.full((neg_edge_index.size(1), ), 0, dtype=torch.float)
        val = torch.cat([pos_val, neg_val], dim=0)

        row, col = edge_index
        edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        val = torch.cat([val, val], dim=0)

        edge_index, val = coalesce(edge_index, val, num_nodes=N)
        val = val - 1

        # Borrowed from:
        # https://github.com/benedekrozemberczki/SGCN/blob/master/src/utils.py
        edge_index = edge_index.detach().numpy()
        val = val.detach().numpy()
        A = scipy.sparse.coo_matrix((val, edge_index), shape=(N, N))
        svd = TruncatedSVD(n_components=self.in_channels, n_iter=128)
        svd.fit(A)
        x = svd.components_.T
        return torch.from_numpy(x).to(torch.float).to(pos_edge_index.device)

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = torch.nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                torch.nn.init.constant_(mask, 0.0)
        return mask

    def mask_feature(self, x):
        feat_mask = torch.sigmoid(self.feat_mask).to(x.device)
        return x * feat_mask

    def encode(self, x, pos_edge_index, neg_edge_index):
        # one conv each epoch
        x = self.mask_feature(x)
        z = x.clone()
        for conv in self.convs:
            z = torch.tanh(conv(z, pos_edge_index, neg_edge_index))
        self.tmp_z = self.weight(z)

        # edge logits whether noise
        pos_logit = (self.tmp_z[pos_edge_index[0]] * self.tmp_z[pos_edge_index[1]]).sum(dim=-1)
        neg_logit = (self.tmp_z[neg_edge_index[0]] * self.tmp_z[neg_edge_index[1]]).sum(dim=-1)
        pos_weight = torch.nn.Sigmoid()(pos_logit)
        neg_weight = torch.nn.Sigmoid()(neg_logit)
        if self.training:
            self.encode_pos_weight = pos_weight
            self.encode_neg_weight = neg_weight

        # edge sampling
        sampled_pos = (pos_weight > self.SAMPLING_RATIO * torch.rand_like(pos_weight)).detach()
        sampled_neg = (neg_weight > self.SAMPLING_RATIO * torch.rand_like(neg_weight)).detach()
        new_pos_index = pos_edge_index[:, sampled_pos]
        new_neg_index = neg_edge_index[:, sampled_neg]
        if self.training:
            self.new_pos_index = new_pos_index
            self.new_neg_index = new_neg_index

        xs = 0
        for conv in self.convs[:-1]:
            x = torch.tanh(conv(x, new_pos_index, new_neg_index))
            xs += x
        x = self.convs[-1](x, new_pos_index, new_neg_index)
        x = self.weight(torch.tanh(x))
        xs += x
        return xs

    def discriminate(self, z, edge_index):
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        return torch.log_softmax(value, dim=1)

    def cls_loss(self, z, pos_edge_index, neg_edge_index, is_sample=True):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        pos_loss = F.nll_loss(self.discriminate(z, pos_edge_index),
            pos_edge_index.new_full((pos_edge_index.size(1), ), 0), reduction='none')
        neg_loss = F.nll_loss(self.discriminate(z, neg_edge_index),
            neg_edge_index.new_full((neg_edge_index.size(1), ), 1), reduction='none')
        none_loss = F.nll_loss(self.discriminate(z, none_edge_index),
            none_edge_index.new_full((none_edge_index.size(1), ), 2), reduction='none')

        if is_sample:
            pos_loss = pos_loss * self.encode_pos_weight
            neg_loss = neg_loss * self.encode_neg_weight
        sup_loss = (pos_loss.mean() + neg_loss.mean() + none_loss.mean())/3
        return sup_loss

    def pos_embedding_loss(self, z, pos_edge_index):
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))
        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z, neg_edge_index):
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))
        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()
