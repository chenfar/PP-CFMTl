# cluster的输入包括所有客户端中上传的梯度参数，客户端index都是秘密共享的形式

import time

import crypten
import torch
import numpy as np

from crypten.mpc import run_multiprocess


def _getname(i, j):
    t = [i, j]
    t.sort()
    return str(t[0]) + '-' + str(t[1])


def _clusters(group, w_local, args):
    # w_local_sharing=ArithmeticSharedTensor(w_local,precision=0)
    X = []
    for i in range(len(w_local)):
        tmp = []
        for k in w_local[i].keys():
            tmp.append(w_local[i][k].flatten())
        X.append(crypten.cat(tmp))
    distance_matrix = [[0 for j in range(args.num_clients)] for i in range(args.num_clients)]
    print(f"begin to compute distance, client num {args.num_clients}")
    for i in range(0, args.num_clients):
        for j in range(i, args.num_clients):
            tmp = (X[i] - X[j]) * (X[i] - X[j])
            distance_matrix[i][j] = tmp.sqrt()
            print("ok......")
    print("distance compute over.....")

    clusterDistance = dict()
    clusterMap = dict()
    clusterCount = args.num_clients - 1
    for i in range(0, args.num_clients - 1):
        for j in range(i, args.num_clients):
            name = _getname(i, j)
            clusterDistance[name] = distance_matrix[i][j]
    for k in range(0, args.num_clients):
        clusterMap[k] = 1
    clusterIndex = [1 for i in range(len(clusterMap))]
    clusterpoint = [[] for i in range(len(clusterMap))]
    for i in range(len(clusterMap)):
        clusterpoint[i].append(i)
    while True:
        if len(clusterMap) == args.clust:
            break
        clusterList = []
        for key in clusterMap:
            clusterList.append(key)
        clusterListLength = len(clusterList)
        now_i = 0
        now_j = 0
        min_distance_val = None
        for i in range(0, clusterListLength - 1):
            for j in range(i + 1, clusterListLength):
                name = _getname(clusterList[i], clusterList[j])
                if (min_distance_val is None or ((min_distance_val > clusterDistance[name]).get_plain_text())[
                    0] == 1) and clusterIndex[clusterList[i]] == 1 and clusterIndex[clusterList[j]] == 1:
                    now_i = i
                    now_j = j
                    min_distance_val = clusterDistance[name]
        now_cluster_i = clusterList[now_i]
        now_cluster_j = clusterList[now_j]
        print(f"chose {now_cluster_i},{now_cluster_j}", )
        ni = clusterMap[now_cluster_i]
        nj = clusterMap[now_cluster_j]
        clusterIndex[now_cluster_i] = 0
        clusterIndex[now_cluster_j] = 0

        del clusterMap[now_cluster_i]
        del clusterMap[now_cluster_j]

        clusterCount += 1
        clusterIndex.append(1)
        clusterMap[clusterCount] = ni + nj
        clusterpoint.append(clusterpoint[now_cluster_i] + clusterpoint[now_cluster_j])
        clusterpoint[now_cluster_i] = []
        clusterpoint[now_cluster_j] = []
        for k in clusterMap.keys():
            if k == clusterCount:
                continue
            else:  # 计算新的距离
                nk = clusterMap[k]
                alpha_i = (ni + nk) / (ni + nj + nk)
                alpha_j = (nj + nk) / (ni + nj + nk)
                beta = -nk / (ni + nj + nk)
                newDistance = alpha_i * clusterDistance[_getname(now_cluster_i, k)]
                newDistance += alpha_j * clusterDistance[_getname(now_cluster_j, k)]
                newDistance += beta * clusterDistance[_getname(now_cluster_i, now_cluster_j)]
                clusterDistance[_getname(clusterCount, k)] = newDistance
    finalcluster = [x for x in clusterpoint if x]

    # build rel
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

    new_groups = []
    for cluster in finalcluster:
        tmp = []
        for c in cluster:
            tmp.append(group[c])
        new_groups.append(tmp)
    return new_groups, one_hot, rel


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
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--clust', type=int, default=5)
    parser.add_argument('--dist', type=str, default='L2')

    from CFMTL.model import Net_mnist
    @run_multiprocess(world_size=2)
    def test(args):
        # w_local = torch.load(f="./w_local.pth")
        w_local = [Net_mnist().state_dict() for i in range(args.num_clients)]
        print(len(w_local))
        from aggre import encrypt_w
        w_local_enc = encrypt_w(w_local)
        simulation_clusters(w_local_enc, args)


    args = parser.parse_args()
    test(args)
