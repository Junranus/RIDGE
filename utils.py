#!/usr/bin/env python
# encoding: utf-8
import torch
import random
import numpy as np
import os.path as osp

def sampling_MI(prob, tau=0.8, reduction='none'):
    prob = prob.clamp(1e-4, 1-1e-4)
    entropy1 = prob * torch.log(prob / tau)
    entropy2 = (1-prob) * torch.log((1-prob) / (1-tau))
    res = entropy1 + entropy2
    if reduction == 'none':
        return res
    elif reduction == 'mean':
        return torch.mean(res)
    elif reduction == 'sum':
        return torch.sum(res)


def load_data_all(rel_path, dataset_name, idx=1):
    with open(osp.join(rel_path, f'{dataset_name}-{idx}_train.txt')) as fp:
        train = np.array([[int(s) for s in l.split()] for l in fp])
    with open(osp.join(rel_path, f'{dataset_name}-{idx}_test.txt')) as fp:
        test = np.array([[int(s) for s in l.split()] for l in fp])
    edgelist = np.concatenate((train, test))
    # remap node id
    nodes = sorted(list(set(np.array(edgelist)[:,:2].flatten())))
    id_map = {n: nodes.index(n) for n in nodes}
    train = [(id_map[a], id_map[b], s) for a, b, s in train]
    test = [(id_map[a], id_map[b], s) for a, b, s in test]
    return np.array(train), np.array(test)


def edge_flip(edges, ratio):
    # random noise
    selected_idx = random.sample(range(len(edges)), int(len(edges)*ratio))
    edges[selected_idx, 2] = -edges[selected_idx, 2]
