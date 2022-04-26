# cluster的输入包括所有客户端中上传的梯度参数，客户端index都是秘密共享的形式
import time

from typing import List

from utils import *


def _getname(i, j):
    t = [i, j]
    t.sort()
    return str(t[0]) + '-' + str(t[1])


def _euclidean_dist(X, num_c):
    clusterDistance = dict()
    distList = []
    s = time.time()
    info(f"begin to compute distance, client num {num_c}")
    for i in range(0, num_c - 1):
        distances = (X - X[i]).square().sum(dim=1).sqrt().cpu()
        for j in range(i + 1, num_c):
            name = _getname(i, j)
            clusterDistance[name] = distances[j]
            distList.append({'i': i, 'j': j, 'dist': distances[j]})

    info(f"compute distance time use : {time.time() - s}")
    return clusterDistance, distList


class Heap:
    def __init__(self, data_list: List[dict]):
        self.data_list = data_list
        for i in range(len(data_list) // 2 - 1, -1, -1):
            self.heapify(i)

    def get_parent_index(self, index):
        # 返回父节点的下标
        if index == 0 or index > len(self.data_list) - 1:
            return None
        else:
            return (index - 1) >> 1

    def swap(self, index_a, index_b):
        # 交换数组中的两个元素
        self.data_list[index_a], self.data_list[index_b] = self.data_list[index_b], self.data_list[index_a]

    def cmp(self, i, j):
        return self.data_list[i]['dist'].gt(self.data_list[j]['dist']).get_plain_text().item() == 1

    def insert(self, data):
        # 先把元素放在最后，然后从后往前依次堆化
        # 这里以大顶堆为例，如果插入元素比父节点大，则交换，直到最后
        self.data_list.append(data)
        index = len(self.data_list) - 1
        parent = self.get_parent_index(index)
        # 循环，直到该元素成为堆顶，或小于父节点（对于大顶堆)
        while parent is not None and self.cmp(parent, index):
            # 交换操作
            self.swap(parent, index)
            index = parent
            parent = self.get_parent_index(parent)

    def top(self):
        # 删除堆顶元素，然后将最后一个元素放在堆顶，再从上往下依次堆化
        remove_data = self.data_list[0]
        self.data_list[0] = self.data_list[-1]
        del self.data_list[-1]

        # 堆化
        self.heapify(0)
        return remove_data

    def heapify(self, index):
        # 从上往下堆化，从index 开始堆化操作 (大顶堆)
        total_index = len(self.data_list) - 1
        while True:
            minvalue_index = index
            if 2 * index + 1 <= total_index and self.cmp(index, 2 * index + 1):
                minvalue_index = 2 * index + 1
            if 2 * index + 2 <= total_index and self.cmp(minvalue_index, 2 * index + 2):
                minvalue_index = 2 * index + 2
            if minvalue_index == index:
                break
            self.swap(index, minvalue_index)
            index = minvalue_index


def _clusters_by_heap(group, w_local, args):
    X = flatten(w_local)
    X = X.cuda()

    clusterDistance, distList = _euclidean_dist(X, args.num_clients)

    s = time.time()
    info("build min heap")
    heap = Heap(distList)

    # 2，初始化集群大小的map
    clusterSize = dict()
    for k in range(0, args.num_clients):
        clusterSize[k] = 1

    clusterLastId = args.num_clients - 1

    clusterPoint = [[i] for i in range(len(clusterSize))]

    info("begin to cluster")
    # 层次聚类过程
    while True:
        if len(clusterSize) == args.clust:
            break
        candidate = heap.top()
        cluster = clusterSize.keys()
        while candidate['i'] not in cluster or candidate['j'] not in cluster:
            candidate = heap.top()

        now_cluster_i = candidate['i']
        now_cluster_j = candidate['j']

        info(f"select {now_cluster_i} and {now_cluster_j}")

        ni = clusterSize[now_cluster_i]
        nj = clusterSize[now_cluster_j]

        del clusterSize[now_cluster_i]
        del clusterSize[now_cluster_j]

        clusterLastId += 1
        clusterSize[clusterLastId] = ni + nj
        clusterPoint.append(clusterPoint[now_cluster_i] + clusterPoint[now_cluster_j])
        clusterPoint[now_cluster_i] = []
        clusterPoint[now_cluster_j] = []
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
                heap.insert({'i': clusterLastId, 'j': k, 'dist': newDistance})

    finalCluster = [x for x in clusterPoint if x]  # 去掉空集群
    info(f"cluster time use : {time.time() - s}")

    new_groups = _build_groups(finalCluster, group)
    s = time.time()
    rel = _build_rel(X, args, finalCluster)
    info(f"build rel time use : {time.time() - s}")
    s = time.time()
    one_hot = _build_onehot(args, finalCluster, group)
    info(f"build one-hot time use : {time.time() - s}")
    return new_groups, one_hot, rel


def _clusters(group, w_local, args):
    # 1.计算距离的map
    X = flatten(w_local)
    # clusterDistance = _euclidean_dist(X, args.num_clients)
    distances = crypten.cryptensor(torch.rand(250, 250))
    clusterDistance = dict()
    num_c = args.num_clients
    print(f"begin to compute distance, client num {num_c}")
    for i in range(0, num_c - 1):
        for j in range(i + 1, num_c):
            name = _getname(i, j)
            clusterDistance[name] = distances[i][j]

    # 2，初始化集群大小的map
    clusterSize = dict()
    for k in range(0, args.num_clients):
        clusterSize[k] = 1

    clusterLastId = args.num_clients - 1

    clusterPoint = [[i] for i in range(len(clusterSize))]
    s = time.time()
    # 层次聚类过程
    while True:
        if len(clusterSize) == args.clust:
            break
        clusterList = list(clusterSize.keys())
        clusterListLength = len(clusterList)
        now_i = 0
        now_j = 0
        min_distance_val = None
        for i in range(0, clusterListLength - 1):
            for j in range(i + 1, clusterListLength):
                name = _getname(clusterList[i], clusterList[j])
                if min_distance_val is None or (min_distance_val > clusterDistance[name]).get_plain_text().item() == 1:
                    now_i = i
                    now_j = j
                    min_distance_val = clusterDistance[name]

        now_cluster_i = clusterList[now_i]
        now_cluster_j = clusterList[now_j]
        print(f"select {now_cluster_i} and {now_cluster_j}")
        ni = clusterSize[now_cluster_i]
        nj = clusterSize[now_cluster_j]

        del clusterSize[now_cluster_i]
        del clusterSize[now_cluster_j]

        clusterLastId += 1
        clusterSize[clusterLastId] = ni + nj
        clusterPoint.append(clusterPoint[now_cluster_i] + clusterPoint[now_cluster_j])
        clusterPoint[now_cluster_i] = []
        clusterPoint[now_cluster_j] = []
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

    finalCluster = [x for x in clusterPoint if x]  # 去掉空集群
    print(time.time() - s)
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
    num_c = args.num_clients

    usertocluter = [0 for _ in range(num_c)]
    for i in range(len(finalcluster)):
        for j in range(len(finalcluster[i])):
            usertocluter[finalcluster[i][j]] = i
    one_hot = crypten.cryptensor(torch.zeros(num_c, args.clust))
    pri_index = crypten.cryptensor(torch.arange(0, num_c))

    for i in range(num_c):
        tmp = (pri_index - group[i]) == 0
        one_hot[:, usertocluter[i]] += tmp
    return one_hot


def _build_rel(X, args, finalcluster):
    X_groups = []
    device = X.device
    for i in range(len(finalcluster)):
        X_group = X.index_select(dim=0, index=torch.tensor(finalcluster[i]).to(device)).sum(dim=0)
        X_groups.append(X_group / len(finalcluster[i]))
    X_groups = crypten.stack(X_groups)

    rel = []
    for i in range(len(finalcluster)):
        if args.dist == 'L2':
            distance = (X_groups - X_groups[i]).norm(dim=1)
            distance = (-1 * distance).exp()
        if args.dist == 'Equal':
            distance = torch.ones(len(finalcluster)) * 0.5
        if args.dist == "L1":
            distance = (X_groups - X_groups[i]).abs().sum(dim=1)
            distance = (-1 * distance).exp()
        if args.dist == "cos":
            X_norm = X_groups.norm(dim=1)
            Xi_norm = X_norm[i]
            distance = 1 - (X_groups * X_groups[i]).sum(dim=1) / (X_norm * Xi_norm)
        tmp = []
        for j in range(len(finalcluster)):
            if i != j:
                tmp.append(distance[j].cpu())
        rel.append(tmp)
    return rel


def simulation_clusters(w_local_enc, args):
    # simulation_shuffle
    group = [i for i in range(args.num_clients)]

    group = np.random.choice(group, args.num_clients, replace=False)
    group_enc = crypten.cryptensor(group)

    group = group_enc.get_plain_text()
    shuffle_w_local = [w_local_enc[int(group[i].item())] for i in range(args.num_clients)]
    return _clusters_by_heap(group_enc, shuffle_w_local, args)
