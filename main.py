import torch

from CFMTL.data import *
from CFMTL.local import Local_Update, Local_Update2
from CFMTL.test import Test
from CFMTL.model import *
import argparse
from aggre import *
import numpy as np
from torchvision import datasets, transforms
from crypten.mpc import multiprocess_wrap

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='cifar')
parser.add_argument('--iid', type=str, default='iid')  # non-iid
parser.add_argument('--ratio', type=float, default=0.25)

parser.add_argument('--method', type=str, default='CFMTL')
parser.add_argument('--ep', type=int, default=100)
parser.add_argument('--local_ep', type=int, default=1)
parser.add_argument('--frac', type=float, default=0.2)
parser.add_argument('--num_batch', type=int, default=10)

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--decay', type=float, default=0)
parser.add_argument('--momentum', type=float, default=0.5)

parser.add_argument('--num_clients', type=int, default=50)
parser.add_argument('--clust', type=int, default=10)
parser.add_argument('--if_clust', type=bool, default=True)

parser.add_argument('--prox', type=bool, default=True)
parser.add_argument('--R', type=str, default='L2')
parser.add_argument('--prox_local_ep', type=int, default=10)
parser.add_argument('--prox_lr', type=float, default=0.01)
parser.add_argument('--prox_momentum', type=float, default=0.5)
parser.add_argument('--L', type=float, default=1)
parser.add_argument('--dist', type=str, default='L2')

parser.add_argument('--experiment', type=str, default='performance-cifar')

import multiprocessing as mp


def save_result():
    record = []
    record.append(copy.deepcopy(acc_final))
    record.append(copy.deepcopy(pro_final))
    record = np.array(record)
    print(record)
    if args.iid == "iid":
        filename = f'experiments/{args.experiment}-iid-secure.npy'
    elif args.iid == "non-iid":
        filename = f'experiments/{args.experiment}-noniid-{args.ratio}-secure.npy'
    else:  # non-iid-single_class
        filename = f'experiments/{args.experiment}-non-iid-single_class-secure.npy'
    np.save(filename, record)

# Clustered Secure Sparse Aggregation for Federated Learning with Non-IID Data
# Secure Sparse Aggregation with hierarchical clustering for Federated Learning on Non-IID Data
if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = parser.parse_args()

    if args.dataset == 'mnist':
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
    else:  # args.dataset == 'cifar'
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar/', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar/', train=False, download=True, transform=trans_cifar)
        if args.iid == 'iid':
            dict_train, dict_test = cifar_iid(dataset_train, dataset_test, args.num_clients, 10)
        elif args.iid == 'non-iid':
            dict_train, dict_test = cifar_non_iid(dataset_train, dataset_test, args.num_clients, 10, args.ratio)
        else:  # args.iid == 'non-iid-single_class':
            dict_train, dict_test = cifar_non_iid_single_class(dataset_train, dataset_test, args.num_clients, 10)
        Net = Net_cifar

    if args.experiment == 'performance-mnist' or args.experiment == 'performance-cifar':
        acc_final = [[] for i in range(3)]
        pro_final = [[] for i in range(3)]
        for m in range(2, 3):
            if m == 0:
                args.method = 'FL'
            if m == 1:
                args.method = 'CFMTL'
                args.prox = False
                args.if_clust = True
            if m == 2:
                args.method = 'CFMTL'
                args.prox = True
                args.if_clust = True

            if args.method == 'FL':
                loss_train = []
                pro_train = pro_final[m]
                acc_train = acc_final[m]
                w_global = Net().state_dict()

                for iter in range(args.ep):
                    w_local = []
                    loss_local = []
                    num_clients = max(int(args.frac * args.num_clients), 1)
                    clients = np.random.choice(range(args.num_clients), num_clients, replace=False)
                    for id in clients:
                        mem, w, loss = Local_Update(args, w_global, dataset_train, dict_train[id], iter)
                        w_local.append(w)
                        loss_local.append(loss)

                    multiprocess_wrap(func=FedAvg, world_size=2, args=(w_local,))
                    w_global = torch.load("./w_avg.pth")

                    loss_avg = sum(loss_local) / len(loss_local)
                    print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
                    loss_train.append(loss_avg)

                    acc_test = []
                    for id in range(args.num_clients):
                        acc, loss_test = Test(args, w_global, dataset_test, dict_test[id])
                        acc_test.append(acc)
                    acc_avg = sum(acc_test) / len(acc_test)
                    print("Testing accuracy: {:.2f}".format(acc_avg))
                    acc_train.append(acc_avg)
                    acc_client_num_95 = 0
                    for acc_client in acc_test:
                        if acc_client >= 95:
                            acc_client_num_95 += 1
                    pro_train.append(acc_client_num_95 / args.num_clients * 100)
                    print("Testing proportion: {:.1f}".format(acc_client_num_95 / args.num_clients * 100))

            if args.method == 'CFMTL':
                groups = [[i for i in range(args.num_clients)]]
                loss_train = []
                pro_train = pro_final[m]
                acc_train = acc_final[m]
                w_groups = [Net().state_dict()]

                for iter in range(args.ep):
                    loss_local = []
                    num_group = len(groups)

                    w_local = [None for i in range(args.num_clients)]  # 所有客户端w的集合torch.save

                    # 并行在客户端执行的
                    for group_id in range(num_group):
                        group = groups[group_id]
                        if iter > 0:
                            num_clients = max(int(args.frac * len(group)), 1)
                            clients = np.random.choice(group, num_clients, replace=False)
                            group = clients
                        w_group = w_groups[group_id]
                        for id in group:
                            mem, w, loss = Local_Update(args, w_group, dataset_train, dict_train[id], iter)
                            w_local[id] = w  # 添加到server端
                            loss_local.append(loss)

                    if iter == 0:
                        torch.save(w_local, './w_local.pth')
                        multiprocess_wrap(Cluster_Init, world_size=2, args=(args,))
                        exit(1)
                        groups, w_groups, _, _ = torch.load("./rank0.pth")

                    else:
                        torch.save(w_local, './w_local_sub.pth')
                        multiprocess_wrap(Cluster_FedAvg, world_size=2, args=(args,))
                        w_groups = torch.load("./w_groups.pth")

                    loss_avg = sum(loss_local) / len(loss_local)
                    print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
                    loss_train.append(loss_avg)

                    if iter == 0:
                        print("Groups Number: ", len(groups))
                        print(groups)

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
                    acc_train.append(acc_avg)
                    acc_client_num_95 = 0
                    for acc_client in acc_test:
                        if acc_client >= 95:
                            acc_client_num_95 += 1
                    pro_train.append(acc_client_num_95 / args.num_clients * 100)
                    print("Testing proportion: {:.1f}".format(acc_client_num_95 / args.num_clients * 100))

            save_result()

# python
