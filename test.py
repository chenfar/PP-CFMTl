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
    parser.add_argument('--num_clients', type=int, default=250)
    parser.add_argument('--clust', type=int, default=50)
    parser.add_argument('--dist', type=str, default='L2')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--prox', type=bool, default=False)
    parser.add_argument('--R', type=str, default='L2')
    parser.add_argument('--prox_local_ep', type=int, default=10)
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
        finalcluster = [[18], [100], [223], [92], [96, 78, 94], [51, 63], [91, 82, 97], [198, 182, 184], [248, 226],
                        [215, 220, 221, 210], [44, 43, 45, 36], [22, 13, 1, 23, 14], [157, 174, 159], [75, 77, 76],
                        [212, 207, 213], [98, 84], [102, 103, 109, 118], [107, 124, 119, 123, 117], [206, 202, 203],
                        [138, 149, 134, 145, 139], [95, 87], [128, 146, 147, 137, 141],
                        [126, 125, 136, 135, 132, 140, 127, 129], [158, 172], [9, 15, 0, 3, 2, 21],
                        [189, 181, 180, 193, 195, 177, 176, 196, 178], [188, 186, 187, 197, 191, 190, 185],
                        [183, 194, 175, 179, 192, 199], [217, 211, 218, 204, 222], [153, 160, 166, 151, 162],
                        [16, 10, 24, 6, 4, 11], [143, 142, 133, 148, 144, 130, 131],
                        [101, 106, 112, 111, 115, 122, 121], [163, 152, 168], [19, 7, 8, 12, 5, 17, 20],
                        [93, 85, 86, 81], [219, 205, 224, 201, 216, 200],
                        [234, 246, 245, 244, 243, 237, 235, 240, 228, 229, 231, 247], [46, 38, 37, 30, 47],
                        [214, 209, 208], [110, 114, 116, 108, 113, 120, 104, 105], [68, 55, 52, 65, 61, 74, 73],
                        [60, 71, 70, 58, 62, 67, 64, 57, 69, 72, 54], [59, 56, 50, 66, 53],
                        [26, 35, 29, 32, 34, 33, 48, 25, 49, 27], [156, 150, 165, 173, 169, 170],
                        [155, 167, 161, 164, 154, 171], [80, 79, 83, 88, 99, 90, 89],
                        [241, 230, 238, 232, 225, 236, 233, 242, 249, 239, 227], [42, 40, 28, 41, 39, 31]]
        w_local = torch.load(f="./w_local.pth")
        from aggre import encrypt_w
        w_local_enc = encrypt_w(w_local)
        X = flatten(w_local_enc)
        X = X.cuda()
        s = time.time()
        rel = _build_rel(X, args, finalcluster)
        info(f"{time.time() - s}")


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
        new_w_groups, args, rel = torch.load(f"./rank{dist.get_rank()}-prox.pth")
        Prox(new_w_groups, args, rel)


    args = parser.parse_args()
    from aggre import Cluster_Init

    # multiprocess_wrap(Cluster_Init, args=(None, args))
    multiprocess_wrap(prox_test, args=())
