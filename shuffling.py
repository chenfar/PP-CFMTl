
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../')))
# print(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../')))
import crypten
from crypten.mpc.ptype import ptype as Ptype
from functools import reduce
import numpy as np
import random
import time
from crypten.mpc import run_multiprocess
import crypten.communicator as comm
import crypten
import torch
from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
from crypten.mpc.primitives.binary import BinarySharedTensor
from crypten.mpc.primitives.converters import convert

import warnings
warnings.filterwarnings("ignore")


@run_multiprocess(world_size=2)
def shuffling():
    client = 4
    labels = crypten.cryptensor(torch.tensor(torch.arange(0, client), dtype=torch.int64))
    ttext =  crypten.cryptensor(torch.tensor([[1,2,3,4,5,6],[2,3,4,5,6,7],[3,4,5,6,7,8],[3,4,5,6,7,8]], dtype=torch.int64))

    rank = comm.get().get_rank()
 
    zeros =  torch.zeros([client,client], dtype=torch.int64)
    one = torch.ones(client, dtype=torch.int64)
    dum = torch.zeros([client, client], dtype=torch.int64)

    one_hots = crypten.cryptensor(zeros)

    c1 = zeros.clone()
    c2 = zeros.clone()

    np.random.seed(int(time.time())+rank)
    shuffling = np.random.permutation(client)
    print("shuffling:", shuffling)
    all_tensor = []

    for item in shuffling:
        if rank == 0:
            row = item
            c1 = zeros.clone()
            c1[row, :] = one 
            all_tensor += [c1]
        else:
            col = item
            c2 = zeros.clone()
            c2[:,col] = one
            all_tensor += [c2]

    if rank == 0:
        a = [BinarySharedTensor.from_shares(i, src = rank, precision=0) for i in all_tensor]
        b =  [BinarySharedTensor.from_shares(dum, src = rank, precision=0) for i in range(len(all_tensor))]
    else:
        a = [BinarySharedTensor.from_shares(dum, src = rank, precision=0) for i in range(len(all_tensor))]
        b = [BinarySharedTensor.from_shares(i, src = rank, precision=0) for i in all_tensor]

    for i, j in zip(a, b):
        k1 = i & j 
        result = convert(k1, Ptype.arithmetic, bits=1)
        one_hots = one_hots + result

    print(one_hots.get_plain_text())
    shufflingout = labels @ one_hots
    shufflingover = one_hots @ ttext 
    print("shuffling label", shufflingout.get_plain_text())
    print("onehotsss:", shufflingover.get_plain_text())


if __name__ == '__main__':
    shuffling()

  