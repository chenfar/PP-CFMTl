import torch

from CFMTL.data import *
from CFMTL.local import Local_Update
from CFMTL.test import Test
from CFMTL.model import *
import argparse
from aggre import *
import numpy as np
from torchvision import datasets, transforms
from crypten.mpc import multiprocess_wrap

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--iid', type=str, default='non-iid')
parser.add_argument('--ratio', type=float, default=0.5)

parser.add_argument('--method', type=str, default='CFMTL')
parser.add_argument('--ep', type=int, default=20)
parser.add_argument('--local_ep', type=int, default=1)
parser.add_argument('--frac', type=float, default=0.3)
parser.add_argument('--num_batch', type=int, default=10)

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--decay', type=float, default=0)
parser.add_argument('--momentum', type=float, default=0.5)

parser.add_argument('--num_clients', type=int, default=20)
parser.add_argument('--clust', type=int, default=5)
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
import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = parser.parse_args()

    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    if args.iid == 'iid':
        dict_train, dict_test = mnist_iid(dataset_train, dataset_test, args.num_clients, 10)
    elif args.iid == 'non-iid':
        dict_train, dict_test = mnist_non_iid(dataset_train, dataset_test, args.num_clients, 10, args.ratio)
    else:  # args.iid == 'non-iid-single_class'
        dict_train, dict_test = mnist_non_iid_single_class(dataset_train, dataset_test, args.num_clients, 10)
    Net = Net_mnist

    groups = [[i for i in range(args.num_clients)]]
    loss_train = []
    pro_train = []
    acc_train = []
    w_groups = [Net().state_dict()]

    for iter in range(args.ep):
        loss_local = []
        num_group = len(groups)

        w_local = [None for i in range(args.num_clients)]  # 所有客户端w的集合torch.save

        # 并行在客户端执行的
        # if iter > 0:
        for group_id in range(num_group):
            group = groups[group_id]
            if iter > 0:
                num_clients = max(int(args.frac * len(group)), 1)
                if iter == 1:
                    print(f"group {group_id} select {num_clients}/{len(group)} client")
                clients = np.random.choice(group, num_clients, replace=False)
                group = clients
            w_group = w_groups[group_id]
            for id in group:
                mem, w, loss = Local_Update(args, w_group, dataset_train, dict_train[id], iter)
                w_local[id] = w  # 添加到server端
                loss_local.append(loss)

        print("client update w_locals, begin to do aggregation")
        torch.save(w_local, "./w_local.pth")
        # w_local = torch.load("./w_local.pth")
        # exit()

        if iter == 0:
            multiprocess_wrap(Cluster_Init, world_size=2, args=(w_local, args,))

            groups, w_groups, rel0, one_hot_share0 = torch.load("./rank0.pth")
            rel1, one_hot_share1 = torch.load("./rank1.pth")

            one_hot_share = [one_hot_share0, one_hot_share1]
            rel = [rel0, rel1]

        else:
            # w_groups = FedAvg(groups,w_local)  明文聚合
            multiprocess_wrap(Cluster_FedAvg, world_size=2, args=(w_local, one_hot_share, rel, args,))
            w_groups = torch.load("./w_groups.pth")

        if iter > 0:
            loss_avg = sum(loss_local) / len(loss_local)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)

        if iter == 0:
            print("Groups Number: ", len(groups))
            print(groups)
            exit()

        acc_test = []
        num_group = len(groups)
        for group_id in range(num_group):
            group = groups[group_id]
            w_group = w_groups[group_id]
            for id in group:
                acc, loss_test = Test(args, w_group, dataset_test, dict_test[id])
                acc_test.append(acc)
        acc_avg = sum(acc_test) / len(acc_test)
        print("Testing accuracy: {:.2f}".format(acc_avg))
        acc_train.append(acc_avg.item())
        acc_client_num_95 = 0
        for acc_client in acc_test:
            if acc_client >= 95:
                acc_client_num_95 += 1
        pro_train.append(acc_client_num_95 / args.num_clients * 100)
        print("Testing proportion: {:.1f}".format(acc_client_num_95 / args.num_clients * 100))
    print(acc_train)
    print(loss_train)
    print(pro_train)
