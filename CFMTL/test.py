import argparse
from model import *
import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset


class DatasetFolder(Dataset):
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        images, labels = self.dataset[self.ids[item]]
        return images, labels


def Test(args, w, dataset=None, ids=None):
    dataloader = DataLoader(DatasetFolder(dataset, ids), batch_size=args.num_batch, shuffle=True)

    if args.dataset == 'mnist':
        Net = Net_mnist
    if args.dataset == 'cifar':
        Net = Net_cifar

    net = Net()
    net.load_state_dict(w)
    net.eval()

    loss_test = 0
    correct = 0

    for data in dataloader:
        images, labels = data

        log_probs = net(images)
        loss_test += F.cross_entropy(log_probs, labels, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

    loss_test /= len(dataloader.dataset)
    acc = 100.00 * correct / len(dataloader.dataset)

    return acc, loss_test


from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist

X = [[0, 0], [0, 1], [1, 0],
     [0, 4], [0, 3], [1, 4],
     [4, 0], [3, 0], [4, 1],
     [4, 4], [3, 4], [4, 3]]

# y = pdist()
# print(y)
from sklearn.cluster import AgglomerativeClustering
print(ward(X))