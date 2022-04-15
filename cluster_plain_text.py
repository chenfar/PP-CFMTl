import time

import torch
from typing import List

clusterDistance = dict()  # 存放cluster之间的距离，形如'1-2':3表示cluster1与cluster2之间的距离为3
clusterMap = dict()  # 存放cluster的情况，形如'1':4表示cluster1里面有4个元素（样本）
clusterCount = 0  # 每合并一次生成新的序号来命名cluster


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
        return self.data_list[i]['dist'] > self.data_list[j]['dist']

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


def ward_linkage_method(distance_matrix):
    N = len(distance_matrix)
    clusterCount = N - 1
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            name = getName(i, j)
            clusterDistance[name] = distance_matrix[i][j]
    for k in range(0, N):
        clusterMap[k] = 1
    s = time.time()
    while True:
        # 查找距离最短的两个cluster
        # clusterDistance里面有冗余（即合并后之前的距离仍在，
        # 所以循环以clusterMap为准，这个里面没有冗余。
        tmp = 100000
        clusterList = list(clusterMap.keys())
        clusterListLength = len(clusterList)
        for iii in range(0, clusterListLength - 1):
            for jjj in range(iii + 1, clusterListLength):
                name = getName(clusterList[iii], clusterList[jjj])
                if tmp > clusterDistance[name]:
                    i = clusterList[iii]
                    j = clusterList[jjj]
                    tmp = clusterDistance[name]
        ni = clusterMap[i]  # 第i个cluster内的元素数
        nj = clusterMap[j]
        del clusterMap[i]  # 删掉原来的cluster
        del clusterMap[j]
        clusterCount += 1  # 新增新的cluster
        clusterMap[clusterCount] = ni + nj  # 新cluster的元素数是之前的总和

        # print(i, j, '->', clusterCount, tmp)  # 输出合并信息:i,j合并为clusterCount，合并高度（距离）为tmp

        if len(clusterMap) == 1: break  # 合并到只剩一个集合为止，然后退出

        # 更新没合并的cluster到新合并后的cluster的距离
        for k in clusterMap.keys():
            if k == clusterCount:
                continue
            else:  # 计算新的距离
                nk = clusterMap[k]
                alpha_i = (ni + nk) / (ni + nj + nk)
                alpha_j = (nj + nk) / (ni + nj + nk)
                beta = -nk / (ni + nj + nk)
                newDistance = alpha_i * clusterDistance[getName(i, k)]
                newDistance += alpha_j * clusterDistance[getName(j, k)]
                newDistance += beta * clusterDistance[getName(i, j)]
                # 把新的距离加入距离集合
                clusterDistance[getName(clusterCount, k)] = newDistance
    print(time.time() - s)


def ward_linkage_method2(distance_matrix):
    N = len(distance_matrix)
    clusterCount = N - 1

    distanceList = []
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            name = getName(i, j)
            clusterDistance[name] = distance_matrix[i][j]
            distanceList.append({'i': i, 'j': j, 'dist': distance_matrix[i][j]})

    heap = Heap(distanceList)

    for k in range(0, N):
        clusterMap[k] = 1
    s = time.time()
    while True:
        # 查找距离最短的两个cluster
        # clusterDistance里面有冗余（即合并后之前的距离仍在，
        # 所以循环以clusterMap为准，这个里面没有冗余。
        clusterList = clusterMap.keys()
        candidate = heap.top()
        while candidate['i'] not in clusterList or candidate['j'] not in clusterList:
            candidate = heap.top()

        i = candidate['i']
        j = candidate['j']

        ni = clusterMap[i]  # 第i个cluster内的元素数
        nj = clusterMap[j]
        del clusterMap[i]  # 删掉原来的cluster
        del clusterMap[j]
        clusterCount += 1  # 新增新的cluster
        clusterMap[clusterCount] = ni + nj  # 新cluster的元素数是之前的总和

        # print(i, j, '->', clusterCount, tmp)  # 输出合并信息:i,j合并为clusterCount，合并高度（距离）为tmp

        if len(clusterMap) == 1: break  # 合并到只剩一个集合为止，然后退出

        # 更新没合并的cluster到新合并后的cluster的距离
        for k in clusterMap.keys():
            if k == clusterCount:
                continue
            else:  # 计算新的距离
                nk = clusterMap[k]
                alpha_i = (ni + nk) / (ni + nj + nk)
                alpha_j = (nj + nk) / (ni + nj + nk)
                beta = -nk / (ni + nj + nk)
                newDistance = alpha_i * clusterDistance[getName(i, k)]
                newDistance += alpha_j * clusterDistance[getName(j, k)]
                newDistance += beta * clusterDistance[getName(i, j)]
                # 把新的距离加入距离集合
                clusterDistance[getName(clusterCount, k)] = newDistance
                heap.insert({'i': clusterCount, 'j': k, 'dist': newDistance})
    print(time.time() - s)


def getName(i, j):
    t = [i, j]
    t.sort()
    return str(t[0]) + '-' + str(t[1])
