# cluster的输入包括所有客户端中上传的梯度参数，客户端index都是秘密共享的形式

import crypten
import torch
import numpy as np

from crypten.mpc import run_multiprocess
from utils import *


def _getname(i, j):
    t = [i, j]
    t.sort()
    return str(t[0]) + '-' + str(t[1])


def _euclidean_dist(X, num_c):
    clusterDistance = dict()
    print(f"begin to compute distance, client num {num_c}")
    for i in range(0, num_c - 1):
        for j in range(i + 1, num_c):
            name = _getname(i, j)
            clusterDistance[name] = (X[i] - X[j]).square().sum().sqrt()

    print("distance compute over.....")
    return clusterDistance


def _clusters(group, w_local, args):
    # 初始化工作
    # 1.计算距离的map
    X = flatten(w_local)
    clusterDistance = _euclidean_dist(X, args.num_clients)
    # 2，初始化集群大小的map
    clusterSize = dict()
    for k in range(0, args.num_clients):
        clusterSize[k] = 1

    clusterLastId = args.num_clients - 1

    clusterIndex = [1 for _ in range(len(clusterSize))]
    clusterpoint = [[] for _ in range(len(clusterSize))]
    for i in range(len(clusterSize)):
        clusterpoint[i].append(i)

    # 层次聚类过程
    while True:
        if len(clusterSize) == args.clust:
            break
        clusterList = []
        for key in clusterSize:
            clusterList.append(key)
        clusterListLength = len(clusterList)
        now_i = 0
        now_j = 0
        min_distance_val = None
        for i in range(0, clusterListLength - 1):
            for j in range(i + 1, clusterListLength):
                name = _getname(clusterList[i], clusterList[j])
                if clusterIndex[clusterList[i]] == 1 and clusterIndex[clusterList[j]] == 1 \
                        and (min_distance_val is None or (
                        min_distance_val > clusterDistance[name]).get_plain_text().item() == 1):
                    now_i = i
                    now_j = j
                    min_distance_val = clusterDistance[name]
        now_cluster_i = clusterList[now_i]
        now_cluster_j = clusterList[now_j]
        ni = clusterSize[now_cluster_i]
        nj = clusterSize[now_cluster_j]
        clusterIndex[now_cluster_i] = 0
        clusterIndex[now_cluster_j] = 0

        del clusterSize[now_cluster_i]
        del clusterSize[now_cluster_j]

        clusterLastId += 1
        clusterIndex.append(1)
        clusterSize[clusterLastId] = ni + nj
        clusterpoint.append(clusterpoint[now_cluster_i] + clusterpoint[now_cluster_j])
        clusterpoint[now_cluster_i] = []
        clusterpoint[now_cluster_j] = []
        for k in clusterSize.keys():
            if k == clusterLastId:
                continue
            else:  # 计算新的距离
                nk = clusterSize[k]
                alpha_i = (ni + nk) / (ni + nj + nk)
                alpha_j = (nj + nk) / (ni + nj + nk)
                beta = -nk / (ni + nj + nk)
                newDistance = alpha_i * clusterDistance[_getname(now_cluster_i, k)]
                newDistance += alpha_j * clusterDistance[_getname(now_cluster_j, k)]
                newDistance += beta * clusterDistance[_getname(now_cluster_i, now_cluster_j)]
                clusterDistance[_getname(clusterLastId, k)] = newDistance

    finalCluster = [x for x in clusterpoint if x]

    # build rel
    rel = _build_rel(X, args, finalCluster)
    # build one-hot
    one_hot = _build_onehot(args, finalCluster, group)
    # build new-groups
    new_groups = _build_groups(finalCluster, group)
    return new_groups, one_hot, rel


def _build_groups(finalcluster, group):
    new_groups = []
    for cluster in finalcluster:
        tmp = []
        for c in cluster:
            tmp.append(group[c])
        new_groups.append(tmp)
    return new_groups


def _build_onehot(args, finalcluster, group):
    usertocluter = [0 for i in range(args.num_clients)]
    for i in range(len(finalcluster)):
        for j in range(len(finalcluster[i])):
            usertocluter[finalcluster[i][j]] = i
    one_hot = crypten.cryptensor(torch.zeros(args.num_clients, args.clust))
    pri_index = []
    for k in range(args.num_clients):
        pri_index.append(crypten.cryptensor(k))
    for i in range(len(usertocluter)):
        for j in range(args.num_clients):
            tmp = (pri_index[j] - group[i])
            one_hot[j][usertocluter[i]] += (tmp == pri_index[0])
    return one_hot


def _build_rel(X, args, finalcluster):
    X_groups = []
    for i in range(len(finalcluster)):
        X_group = X[finalcluster[i][0]]
        for j in range(1, len(finalcluster[i])):
            X_group += X[finalcluster[i][j]]
        X_groups.append(X_group / len(finalcluster[i]))
    rel = []
    for i in range(len(X_groups)):
        rel.append([])
        for j in range(len(X_groups)):
            if j != i:
                if args.dist == 'L2':
                    dist = (X_groups[i] - X_groups[j]).norm()
                    rel[-1].append((-1 * dist).exp())
                if args.dist == 'Equal':
                    rel[-1].append(0.5)
                if args.dist == 'L1':
                    dist = np.sum(((X_groups[i] - X_groups[j]) * (X_groups[i] - X_groups[j])).sqrt())
                    rel[-1].append((-1 * dist).exp())
                if args.dist == 'cos':
                    a = (X_groups[i]).norm()
                    b = (X_groups[j]).norm()
                    dist = 1 - X_groups[i].dot(X_groups[j].T).div(a * b)
                    rel[-1].append(dist)
    return rel


def simulation_clusters(w_local_enc, args):
    # simulation_shuffle
    group = [i for i in range(args.num_clients)]

    group = np.random.choice(group, args.num_clients, replace=False)
    group_enc = crypten.cryptensor(group)

    group = group_enc.get_plain_text()
    shuffle_w_local = [w_local_enc[int(group[i].item())] for i in range(args.num_clients)]
    return _clusters(group_enc, shuffle_w_local, args)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--clust', type=int, default=5)
    parser.add_argument('--dist', type=str, default='L2')

    from CFMTL.model import Net_mnist


    @run_multiprocess(world_size=2)
    def test(args):
        w_local = torch.load(f="./w_local.pth")
        # w_local = [Net_mnist().state_dict() for i in range(args.num_clients)]
        print(len(w_local))
        from aggre import encrypt_w
        w_local_enc = encrypt_w(w_local)
        new_groups, one_hot, rel = simulation_clusters(w_local_enc, args)
        for i in range(len(new_groups)):
            for j in range(len(new_groups[i])):
                new_groups[i][j] = int(new_groups[i][j].get_plain_text().item())
        print(new_groups)
        # [[6], [7, 17], [15, 11, 12, 14, 13, 18], [19, 10, 8], [3, 16, 4, 9, 1, 5, 2, 0]]
        # [[6], [7, 17], [11, 18, 13, 12, 15, 14], [10, 19, 8], [16, 3, 5, 4, 2, 9, 1, 0]]

        # [[2, 3], [14, 15], [0, 1, 16, 17], [8, 9, 12, 13, 18, 19], [4, 5, 6, 7, 10, 11]]
        # [[2, 3], [14, 15], [0, 1, 16, 17], [13, 12, 19, 18, 8, 9], [11, 10, 6, 7, 4, 5]]


    args = parser.parse_args()
    test(args)
