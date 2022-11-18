import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from numpy.linalg import eig, eigh
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, r2_score
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

################################################# model ################################################

class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        # input:  [N]
        # output: [N, d]

        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(1) * div
        eeig = torch.cat((e.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)

        return self.eig_w(eeig)


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class STLayer(nn.Module):

    def __init__(self, nbases, ncombines, prop_dropout=0.0):
        super(STLayer, self).__init__()
        self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        self.prop_dropout = nn.Dropout(prop_dropout)

    def forward(self, x):
        x = self.prop_dropout(x) * self.weight      # [N, m, d] * [1, m, d]
        x = torch.sum(x, dim=1)
        return x


class TransLayer(nn.Module):

    def __init__(self, hidden_dim, nheads, nbases, ncombines, tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0):
        super(TransLayer, self).__init__()

        self.nheads = nheads
        self.hidden_dim = hidden_dim

        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)

        self.decoder = nn.Linear(hidden_dim, nheads)

        self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        self.prop_dropout = nn.Dropout(prop_dropout)

    def forward(self, eig, u, h):
        mha_eig = self.mha_norm(eig)
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig)
        eig = eig + self.mha_dropout(mha_eig) 

        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)

        new_e = self.decoder(eig)   # [N, m]

        basic_feats = [h]
        utx = u.permute(1, 0) @ h
        for i in range(self.nheads):
            basic_feats.append(u @ (new_e[:, i].unsqueeze(1) * utx))  # [N, d]
        basic_feats = torch.stack(basic_feats, axis=1)                # [N, m, d]

        x = self.prop_dropout(basic_feats) * self.weight      # [N, m, d] * [1, m, d]
        x = torch.sum(x, dim=1)
        return eig, x

    
class Specformer(nn.Module):

    def __init__(self, nclass, nfeat, nlayer, hidden_dim=128, nheads=4, tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0):
        super(Specformer, self).__init__()

        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        
        self.feat_encoder = nn.Sequential(
            nn.Linear(self.nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
        )
        self.eig_w = nn.Linear(1, hidden_dim)
        self.eig_encoder = SineEncoding(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, nheads)

        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)
        self.layers = nn.ModuleList([STLayer(nheads+1, nclass, prop_dropout) for i in range(nlayer)])
        
    def forward(self, e, u, x):
        N = e.size(0)
        ut = u.permute(1, 0)

        h = self.feat_dp1(x)
        h = self.feat_encoder(h)
        h = self.feat_dp2(h)

        # ablation
        #eig = self.eig_w(e.unsqueeze(1))
        #mask = torch.BoolTensor([1 if i > 0 else 0 for i in range(N)]).to(e.device)

        raw_eig = eig = self.eig_encoder(e)   # [N, d]

        mha_eig = self.mha_norm(eig)
        #mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig, key_padding_mask=mask)
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig)
        eig = eig + self.mha_dropout(mha_eig) 

        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)

        new_e = self.decoder(eig)   # [N, m]

        '''
        for conv in self.layers:
            eig, h = conv(eig, u, h)
        '''

        ''' ablation
        for _ in range(self.nlayer):
            basic_feats = [h]
            utx = ut @ h
            for i in range(self.nheads):
                basic_feats.append(u @ (new_e[:, i].unsqueeze(1) * utx))  # [N, d]
            basic_feats = torch.stack(basic_feats, axis=1)                # [N, m, d]
            h = self.conv(basic_feats)
        '''

        for conv in self.layers:
            basic_feats = [h]
            utx = ut @ h
            for i in range(self.nheads):
                basic_feats.append(u @ (new_e[:, i].unsqueeze(1) * utx))  # [N, d]
            basic_feats = torch.stack(basic_feats, axis=1)                # [N, m, d]
            h = conv(basic_feats)

        return h, new_e, attn, eig, raw_eig

