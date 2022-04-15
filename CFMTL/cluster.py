import argparse

import torch

from CFMTL.fedavg import FedAvg
import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage,ward
import copy
import math

def Cluster(group, w_local, args):
    X = [[] for i in range(len(group))]
    for i in range(len(group)):
        for j in w_local[i].keys():
            X[i] += w_local[i][j].numpy().flatten().tolist()
    X = np.array(X)
    Z = linkage(X, 'ward')
    if args.if_clust == True:
        clusters = fcluster(Z, args.clust, criterion='maxclust')
    else:
        clusters = np.array(range(1, args.num_clients+1))
    new_groups = [[] for i in range(max(clusters))]
    new_w_groups = []
    for i in range(len(group)):
        new_groups[clusters[i] - 1].append(group[i])
    X_groups = []

    for i in range(max(clusters)):
        new_w_local = []
        X_local = []
        for j in range(len(clusters)):
            if clusters[j] == i+1:
                new_w_local.append(w_local[j])
                X_local.append(X[j])
        new_w_group = FedAvg(new_w_local)
        new_w_groups.append(new_w_group)
        X_group = X_local[0]
        for j in range(1, len(X_local)):
            X_group += X_local[j]
        X_group /= len(X_local)
        X_groups.append(copy.deepcopy(X_group))
    rel = []
    for i in range(len(X_groups)):
        rel.append([])
        for j in range(len(X_groups)):
            if j != i:
                if args.dist == 'L2':
                    dist = np.linalg.norm(X_groups[i]-X_groups[j])
                    rel[-1].append(math.exp(-1*dist))
                if args.dist == 'Equal':
                    rel[-1].append(0.5)
                if args.dist == 'L1':
                    dist = np.sum(np.abs(X_groups[i]-X_groups[j]))
                    rel[-1].append(math.exp(-1*dist))
                if args.dist == 'cos':
                    a = np.linalg.norm(X_groups[i])
                    b = np.linalg.norm(X_groups[j])
                    dist = 1 - np.dot(X_groups[i], X_groups[j].T)/(a * b)
                    rel[-1].append(dist)
    return new_groups, new_w_groups, rel

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--if_clust', type=bool, default=True)
# parser.add_argument('--num_clients', type=int, default=50)
# parser.add_argument('--clust', type=int, default=10)
# parser.add_argument('--dist', type=str, default='L2')
# w_local = torch.load('../w_local.pth')
# group = [i for i in range(len(w_local))]
# args = parser.parse_args()
# new_groups, new_w_groups, rel = Cluster(group, w_local, args)
# print(new_groups)