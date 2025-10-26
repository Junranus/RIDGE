#!/usr/bin/env python
# encoding: utf-8

import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch_geometric
from utils import edge_flip
from utils import sampling_MI
from utils import load_data_all
import torch.nn.functional as F
from model.SGCN_ridge import SGCN
from model.SNEA_ridge import SNEA
from sklearn.metrics import f1_score, roc_auc_score


def train_model(model, optimizer, epoch, x,
        train_pos_edge_index, train_neg_edge_index, args):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index, train_neg_edge_index)

    # loss1: classification loss
    sup_loss = model.cls_loss(z, train_pos_edge_index, train_neg_edge_index)
    loss_1 = model.pos_embedding_loss(z, model.new_pos_index)
    loss_2 = model.neg_embedding_loss(z, model.new_neg_index)
    sup_loss = sup_loss + model.lamb * (loss_1 + loss_2)

    # loss2: I(Y_c; Y)
    encode_edge_weight = torch.cat([model.encode_pos_weight, model.encode_neg_weight])
    KL_Y = sampling_MI(encode_edge_weight, reduction='mean')

    # loss3: I(H; G)
    IB_size = args.hidden_dim//2
    mu = z[:, :IB_size]
    std = F.softplus(z[:, IB_size:]-IB_size, beta=1)
    KL_G = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))

    loss = sup_loss + args.alpha*KL_Y + args.beta*KL_G
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test_model(model, x, train_pos_edge_index, train_neg_edge_index,
        test_pos_edge_index, test_neg_edge_index):
    model.eval()
    z = model.encode(x, train_pos_edge_index, train_neg_edge_index)
    pos_p = model.discriminate(z, test_pos_edge_index)[:, :2].max(dim=1)[1]
    neg_p = model.discriminate(z, test_neg_edge_index)[:, :2].max(dim=1)[1]
    pred = (1 - torch.cat([pos_p, neg_p])).cpu()
    y = torch.cat([pred.new_ones((pos_p.size(0))), pred.new_zeros(neg_p.size(0))])
    pred, y = pred.numpy(), y.numpy()

    auc = roc_auc_score(y, pred)
    f1 = f1_score(y, pred, average='binary') if pred.sum() > 0 else 0
    return auc, f1


def main(args, device):
    aucs, f1s = [], []
    for idx in range(args.repeat_times):
        torch_geometric.seed_everything(42)
        # prepare dataset
        train, test = load_data_all(args.data_dir, args.dataset, idx)
        node_count = max(train.max(), test.max()) + 1

        if args.noise_ratio > 0:
            edge_flip(train, args.noise_ratio)

        train_pos_mask = train[:,2]>0
        train_neg_mask = train[:,2]<0
        test_pos_mask = test[:,2]>0
        test_neg_mask = test[:,2]<0
        train_pos_edge_index = torch.from_numpy(train[train_pos_mask ,0:2].T).to(device).long()
        train_neg_edge_index = torch.from_numpy(train[train_neg_mask ,0:2].T).to(device).long()
        test_pos_edge_index = torch.from_numpy(test[test_pos_mask ,0:2].T).to(device).long()
        test_neg_edge_index = torch.from_numpy(test[test_neg_mask ,0:2].T).to(device).long()

        # prepare model
        if args.gnn_model == 'SGCN':
            model = SGCN(in_channels=args.hidden_dim, hidden_channels=args.hidden_dim,
                num_layers=args.num_gnn_layers, node_num=node_count, lamb=5).to(device)
        elif args.gnn_model == 'SNEA':
            model = SNEA(in_channels=args.hidden_dim, hidden_channels=args.hidden_dim,
                num_layers=args.num_gnn_layers, node_num=node_count, lamb=4).to(device)
        else:
            print('Not supported SGNN encoder!!!')
            return
        x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index, num_nodes=node_count)
        x = x.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learn_rate)

        # train & test
        best_auc = best_f1 = 0
        for epoch in tqdm(range(MAX_EPOCH), ncols=50):
            train_model(model, optimizer, epoch, x, train_pos_edge_index, train_neg_edge_index, args)
            auc, f1 = test_model(model, x, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index)
            if auc > best_auc:
            # if auc + f1 > best_auc +best_f1:
                best_auc = auc
                best_f1 = f1
        print(idx, '-'*10, best_auc, best_f1, flush=True)
        aucs.append(best_auc)
        f1s.append(best_f1)
    print(np.mean(aucs), np.std(aucs), np.mean(f1s), np.std(f1s))


if __name__ == '__main__':
    ### Parse args ###
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='Bitcoin-Alpha')
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--repeat_times', type=int, default=5)
    parser.add_argument('--noise_type', type=str, default='flip')
    parser.add_argument('--gnn_model', type=str, default='SGCN')
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('-r', '--noise_ratio', type=float, default=0.2)
    parser.add_argument('--learn_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--alpha', type=float, default=10, help='coef for KL Y')
    parser.add_argument('--beta', type=float, default=10, help='coef for KL G')

    args = parser.parse_args()

    MAX_EPOCH = 1000
    device = torch.device('cuda:0')
    main(args, device)
