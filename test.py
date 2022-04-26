import time

from cluster import _build_rel
from cluster import *
from aggre import *
from CFMTL.model import *
import torch

# for test
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=200)
    parser.add_argument('--clust', type=int, default=20)
    parser.add_argument('--dist', type=str, default='L2')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--prox', type=bool, default=False)
    parser.add_argument('--R', type=str, default='L2')
    parser.add_argument('--prox_local_ep', type=int, default=5)
    parser.add_argument('--prox_lr', type=float, default=0.01)
    parser.add_argument('--prox_momentum', type=float, default=0.5)
    parser.add_argument('--L', type=float, default=0.1)

    from crypten.mpc import multiprocess_wrap

    from crypten import mpc


    def _build_onehot_test(args):
        # group = [i for i in range(args.num_clients)]
        #
        # group = np.random.choice(group, args.num_clients, replace=False)
        # group_enc = crypten.cryptensor(group)
        w_local = torch.load(f="./w_local.pth")
        from aggre import encrypt_w
        w_local_enc = encrypt_w(w_local)
        simulation_clusters(w_local_enc,args)
        # X = flatten(w_local_enc)
        # X = X.cuda()
        # s = time.time()
        # rel = _build_rel(X, args, finalcluster)
        # info(f"{time.time() - s}")


    def wtest(args):
        mpc.set_activate_protocol("FSS")
        w_local = torch.load(f="./w_local_test.pth")
        # w_local = [Net_mnist().state_dict() for i in range(args.num_clients)]
        info(len(w_local))
        from aggre import encrypt_w
        w_local_enc = encrypt_w(w_local)
        new_groups, one_hot, rel = simulation_clusters(w_local_enc, args)
        torch.set_printoptions(threshold=np.inf)
        info(one_hot.get_plain_text())
        for i in range(len(new_groups)):
            for j in range(len(new_groups[i])):
                new_groups[i][j] = int(new_groups[i][j].get_plain_text().item())
        info(new_groups)


    def cluster_avg_w_test():
        rank = dist.get_rank()
        if rank == 0:
            new_groups, new_w_groups, rel, one_hot = torch.load("./rank0.pth")
        else:
            rel, one_hot = torch.load("./rank1.pth")
        w_local = torch.load('./w_local.pth')
        w_local_enc = encrypt_w(w_local)
        s = time.time()
        new_w_groups = cluster_avg_w(one_hot, w_local_enc)
        info(f"avg time use {time.time() - s}")
        # w_groups = decrypt_w(new_w_groups)
        return rel, new_w_groups


    def prox_test():
        from prox_better import Prox
        new_w_groups, args, rel = torch.load(f"./rank{dist.get_rank()}-prox.pth")
        s = time.time()
        Prox(new_w_groups, args, rel)
        info(time.time() - s)


    args = parser.parse_args()
    from aggre import Cluster_Init

    # multiprocess_wrap(Cluster_Init, args=(None, args))
    multiprocess_wrap(_build_onehot_test, args=(args,))
