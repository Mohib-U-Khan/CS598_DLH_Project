import dhg
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add, scatter
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional
from collections import defaultdict, Counter
from itertools import combinations
from utils import *


# UniGINConv (from UniGNN).
class UniGINConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout=0.):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = nn.Parameter(torch.Tensor([0.1]))

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        Xve = X[vertex]
        Xe = scatter(Xve, edges, dim=0, reduce='mean')
        Xev = Xe[edges]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)
        X = (1 + self.eps) * X + Xv
        X = self.W(X)
        return X


# UniSAGEConv (from UniGNN).
class UniSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout=0.):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        X = self.W(X)
        Xve = X[vertex]
        Xe = scatter(Xve, edges, dim=0, reduce='mean')
        Xev = Xe[edges]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)
        X = X + Xv
        return X


# UniGATConv (from UniGNN).
class UniGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout=0.):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_v = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_drop = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU()
        self.reset_parameters()

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)

    def reset_parameters(self):
        glorot(self.att_v)
        glorot(self.att_e)

    def forward(self, X, vertex, edges):
        H, C, N = self.heads, self.out_channels, X.shape[0]
        X0 = self.W(X)
        X = X0.view(N, H, C)
        Xve = X[vertex]
        Xe = scatter(Xve, edges, dim=0, reduce='mean')
        alpha_e = (Xe * self.att_e).sum(-1)
        a_ev = alpha_e[edges]
        alpha = a_ev
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, vertex, num_nodes=N)
        alpha = self.attn_drop(alpha)
        alpha = alpha.unsqueeze(-1)
        Xev = Xe[edges]
        Xev = Xev * alpha
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)
        X = Xv
        X = X.view(N, H * C)
        X = X + X0
        return X


# UniGCNConv (from UniGNN).
class UniGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout=0.):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)

    def forward(self, X, vertex, edges, H):
        N = X.shape[0]
        degV = torch.sum(H, 1).pow(-0.5)
        degV[torch.isinf(degV)] = 1.0
        degV = torch.unsqueeze(degV, 1)
        degE = torch.sum(H, 0).pow(-0.5)
        degE = torch.unsqueeze(degE, 1)
        X = self.W(X)
        Xve = X[vertex]
        Xe = scatter(Xve, edges, dim=0, reduce='mean')
        Xe = Xe * degE
        Xev = Xe[edges]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)
        Xv = Xv * degV
        X = Xv
        return X


# UniGCNIIConv (from UniGNN).
class UniGCNIIConv(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout=0.):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)

    def forward(self, X, vertex, edges, X0, H):
        alpha = 0.1
        beta = 0.8
        N = X.shape[0]
        degV = torch.sum(H, 1).pow(-0.5)
        degV[torch.isinf(degV)] = 1.0
        degV = torch.unsqueeze(degV, 1)
        degE = torch.sum(H, 0).pow(-0.5)
        degE = torch.unsqueeze(degE, 1)
        Xve = X[vertex]
        Xe = scatter(Xve, edges, dim=0, reduce='mean')
        Xe = Xe * degE
        Xev = Xe[edges]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)
        Xv = Xv * degV
        X = Xv
        Xi = (1 - alpha) * X + alpha * X0
        X = (1 - beta) * Xi + beta * self.W(Xi)
        return X



# AllDeepSets & AllSetTransformer.
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.0, normalization='ln', input_norm=True):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = input_norm
        if normalization == 'bn':
            if num_layers == 1:
                if input_norm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if input_norm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif normalization == 'ln':
            if num_layers == 1:
                if input_norm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if input_norm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout_layer = nn.Dropout(p=dropout)


    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = self.dropout_layer(x)
        x = self.lins[-1](x)
        return x


class PMA(MessagePassing):
    _alpha: OptTensor
    def __init__(self, in_channels, hid_dim,
                 out_channels, num_layers, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, bias=False, **kwargs):
        super(PMA, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout_layer = nn.Dropout(p=dropout)
        self.aggr = 'add'
        self.lin_K = nn.Linear(in_channels, self.heads*self.hidden)
        # For neighbor nodes (source side, value).
        self.lin_V = nn.Linear(in_channels, self.heads*self.hidden)
        self.att_r = nn.Parameter(torch.Tensor(1, heads, self.hidden))
        self.rFF = MLP(in_channels=self.heads*self.hidden,
                       hidden_channels=self.heads*self.hidden,
                       out_channels=out_channels,
                       num_layers=num_layers,
                       dropout=dropout, normalization='None', )
        self.ln0 = nn.LayerNorm(self.heads*self.hidden)
        self.ln1 = nn.LayerNorm(self.heads*self.hidden)
        self.register_parameter('bias', None)
        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index: Adj, size: Size = None, return_attention_weights=None):
        H, C = self.heads, self.hidden
        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_K = self.lin_K(x).view(-1, H, C)
            x_V = self.lin_V(x).view(-1, H, C)
            alpha_r = (x_K * self.att_r).sum(dim=-1)
        out = self.propagate(edge_index, x=x_V, alpha=alpha_r, aggr=self.aggr)
        alpha = self._alpha
        self._alpha = None
        out += self.att_r
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        out = self.ln1(out+F.relu(self.rFF(out)))
        if isinstance(return_attention_weights, bool):
            assert alpha != None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j, alpha_j, index, ptr, size_j):
        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, index.max()+1)
        self._alpha = alpha
        alpha = self.dropout_layer(alpha)
        return x_j * alpha.unsqueeze(-1)

    def aggregate(self, inputs, index, dim_size=None, aggr=None):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class HalfNLHconv(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout,
                 aggr,
                 heads,
                 attention
                 ):
        super(HalfNLHconv, self).__init__()
        self.attention = attention
        self.dropout_layer = nn.Dropout(p=dropout)
        self.aggr = aggr
        if self.attention:
            self.prop = PMA(in_dim, hid_dim, out_dim, num_layers, heads=heads)
        else:
            if num_layers > 0:
                self.f_enc = MLP(in_dim, hid_dim, hid_dim, num_layers, dropout, 'ln', True)
                self.f_dec = MLP(hid_dim, hid_dim, out_dim, num_layers, dropout, 'ln', True)
            else:
                self.f_enc = nn.Identity()
                self.f_dec = nn.Identity()

    def reset_parameters(self):
        if self.attention:
            self.prop.reset_parameters()
        else:
            if not (self.f_enc.__class__.__name__ == 'Identity'):
                self.f_enc.reset_parameters()
            if not (self.f_dec.__class__.__name__ == 'Identity'):
                self.f_dec.reset_parameters()

    def forward(self, x, V, E):
        edge_index = torch.vstack((V, E))
        norm = torch.ones_like(edge_index[0])
        if self.attention:
            x = self.prop(x, edge_index)
        else:
            x = self.dropout_layer(F.relu(self.f_enc(x)))
            x = self.propagate(edge_index, x=x, norm=norm, aggr=self.aggr)
            x = self.f_dec(x)

        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size=None, aggr=None):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)


# AllDeepSets: PMA = False & aggr = 'add'.
# AllSetTransformer: PMA = True & aggr = 'mean'.
class AllSet(nn.Module):
    def __init__(self, in_channels, out_channels, heads, aggr, PMA, device, dropout=0.):
        super().__init__()
        self.V2E = HalfNLHconv(in_channels, out_channels, out_channels, 2, dropout, aggr, heads, PMA)
        self.E2V = HalfNLHconv(out_channels, out_channels, out_channels, 2, dropout, aggr, heads, PMA)
        self.anchor = torch.ones(1, dtype=torch.int16, device=device)
        self.dropout_layer = nn.Dropout(p=dropout)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        v_ext = self.anchor * (X.shape[0] - 1)
        vertex = torch.cat((vertex, v_ext))
        e_ext = torch.max(edges) + self.anchor
        edges = torch.cat((edges, e_ext))
        X = self.V2E(X, vertex, edges)
        X = self.dropout_layer(F.relu(X))
        X = self.E2V(X, edges, vertex)
        return X


# HyperGCNConv.
class HyperGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, is_last, device, dropout=0.):
        super().__init__()
        self.device = device
        self.hypergcn = dhg.nn.HyperGCNConv(in_channels, out_channels, use_mediator=True, drop_rate=dropout, is_last=is_last)

    def forward(self, X, vertex, edges):
        e_list = [tuple(vertex[torch.where(edges == idx)[0]].tolist()) for idx in range(torch.max(edges) + 1)]
        for i, ee in enumerate(e_list):
            if len(ee) == 1:
                e_list[i] = e_list[i] + (1314, 3473)
            elif len(ee) == 2:
                e_list[i] = e_list[i] + (3473,)
            else:
                e_list[i] = e_list[i]
        hg = dhg.Hypergraph(X.shape[0], e_list, device=self.device)
        return self.hypergcn(X, hg)

