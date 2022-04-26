import numpy as np
import torch

import crypten
from crypten import mpc
import torch.distributed as dist

from crypten.mpc import multiprocess_wrap


def flatten(w_local):
    X = []
    for i in range(len(w_local)):
        tmp = []
        for k in w_local[i].keys():
            tmp.append(w_local[i][k].flatten())
        X.append(crypten.cat(tmp))
    return crypten.stack(X)


def info(msg):
    if dist.get_rank() == 0:
        print(msg)


def fss_protocol_test2():
    crypten.cryptensor(a)


if __name__ == '__main__':
    a = torch.ones(5,6)
    print((a*a[1]).sum(dim=1))
    # torch.set_printoptions(threshold=np.inf)
    # print(torch.zeros(250, 250))
    # print(torch.zeros(3,2))
    # mpc.set_activate_protocol("FSS")
    # multiprocess_wrap(func=fss_protocol_test2)
    # print(torch.index_select(torch.ones(5, 5), dim=0, index=torch.tensor([2, 3])).sum(dim=0))
    # a = torch.ones(5, 6)
    # print(a.zero_().inde)
    # print(a.nonzero().view(-1).view(-1,1))
    # b = torch.ones(1,5)*4
    # print(a.sum(dim=1)[2])
