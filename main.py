import time

import torch
from aggre import cluster_avg
# from CFMTL.model import *
from crypten.mpc import run_multiprocess
import crypten
import warnings
import crypten.nn as nn
from prox import Prox

warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--iid', type=str, default='non-iid')
parser.add_argument('--ratio', type=float, default=0.5)

parser.add_argument('--method', type=str, default='CFMTL')
parser.add_argument('--ep', type=int, default=50)
parser.add_argument('--local_ep', type=int, default=1)
parser.add_argument('--frac', type=float, default=0.2)
parser.add_argument('--num_batch', type=int, default=10)

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--decay', type=float, default=0)
parser.add_argument('--momentum', type=float, default=0.5)

parser.add_argument('--num_clients', type=int, default=250)
parser.add_argument('--clust', type=int, default=50)
parser.add_argument('--if_clust', type=bool, default=True)

parser.add_argument('--prox', type=bool, default=True)
parser.add_argument('--R', type=str, default='L2')
parser.add_argument('--prox_local_ep', type=int, default=10)
parser.add_argument('--prox_lr', type=float, default=0.01)
parser.add_argument('--prox_momentum', type=float, default=0.5)
parser.add_argument('--L', type=float, default=0.1)
parser.add_argument('--dist', type=str, default='L2')

parser.add_argument('--experiment', type=str, default='performance-mnist')
parser.add_argument('--filename', type=str, default='fig')

args = parser.parse_args()


@run_multiprocess(world_size=2)
def test():
    one_hot = crypten.cryptensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])
    W = []

    for i in range(7):
        net = Net_mnist()
        net.encrypt()
        W.append(net.state_dict())

    new_w_groups = cluster_avg(one_hot, W)
    rel = []
    # for i in range(len(X_groups)):
    #     rel.append([])
    #     for j in range(len(X_groups)):
    #         if j != i:
    #             if args.dist == 'L2':
    #                 dist = np.linalg.norm(X_groups[i] - X_groups[j])
    #                 rel[-1].append(math.exp(-1 * dist))
    Prox(new_w_groups, args, rel)


from model import *


@run_multiprocess(world_size=2)
def test2():
    net = Net_mnist()
    c_net = net.encrypt()
    s = {}
    for k in c_net.state_dict():
        s[k] = c_net.state_dict()[k]
    new = Net_mnist()
    new.encrypt()
    copy_dict = new.state_dict().copy()
    s.set("_metadata", {'version': 1})
    print(getattr(s, "_metadata", None))

    # new_dict = new.state_dict()
    # import torch.distributed as dist
    # for k in new_dict.keys():
    #     if dist.get_rank() == 0:
    #         print(copy_dict[k])
    #         print("="*10)
    #         print(new_dict[k])
    #     break


# test2()
test()
