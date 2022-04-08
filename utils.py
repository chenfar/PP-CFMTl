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


from crypten.mpc.ptype import ptype as Ptype


@run_multiprocess(world_size=2)
def test():
    a = crypten.cryptensor(torch.ones(2, 2) * 4)
    b = crypten.cryptensor(torch.ones(2, 2))
    t1 = a.to(ptype=Ptype.binary)
    t2 = b.to(ptype=Ptype.binary)
    t1._tensor = (t1._tensor ^ t2._tensor)
    tag = crypten.cryptensor(torch.ones(2, 2) * 5)
    t = t1.to(ptype=Ptype.arithmetic)
    print((tag.eq(t)).get_plain_text())


from CFMTL.cluster import Cluster
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--if_clust', type=bool, default=True)
parser.add_argument('--num_clients', type=int, default=20)
parser.add_argument('--clust', type=int, default=5)
parser.add_argument('--dist', type=str, default='L2')
args = parser.parse_args()
group = [i for i in range(20)]
w_local = torch.load("./w_local.pth")
new_groups, new_w_groups, rel = Cluster(group, w_local, args)
print(new_groups)
