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


test()
