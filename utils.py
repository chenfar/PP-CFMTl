
import time

import torch

import crypten
from crypten.mpc import run_multiprocess

def flatten(w_local):
    X = []
    for i in range(len(w_local)):
        tmp = []
        for k in w_local[i].keys():
            tmp.append(w_local[i][k].flatten())
        X.append(crypten.cat(tmp))
    return X


@run_multiprocess(world_size=2)
def fss_protocol_test2():
    a = crypten.cryptensor(torch.zeros(1))
    b = crypten.cryptensor(torch.ones(1))
    for i in range(10):
        t = time.time()
        t3 = (a.gt(b)).get_plain_text()
        print(time.time() - t)

if __name__ == '__main__':
    fss_protocol_test2()
