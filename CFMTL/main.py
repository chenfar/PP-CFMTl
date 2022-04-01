import argparse
from cluster import Cluster
from data import *
from fedavg import FedAvg
from local import Local_Update
from model import *
import numpy as np
from prox import Prox
from test import Test
import torch
from torchvision import datasets, transforms
import copy

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

if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.iid == 'iid':
            dict_train, dict_test = mnist_iid(dataset_train, dataset_test, args.num_clients, 10)
        elif args.iid == 'non-iid':
            dict_train, dict_test = mnist_non_iid(dataset_train, dataset_test, args.num_clients, 10, args.ratio)
        elif args.iid == 'non-iid-single_class':
            dict_train, dict_test = mnist_non_iid_single_class(dataset_train, dataset_test, args.num_clients, 10)
        Net = Net_mnist

    if args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid == 'iid':
            dict_train, dict_test = cifar_iid(dataset_train, dataset_test, args.num_clients, 10)
        elif args.iid == 'non-iid':
            dict_train, dict_test = cifar_non_iid(dataset_train, dataset_test, args.num_clients, 10, args.ratio)
        elif args.iid == 'non-iid-single_class':
            dict_train, dict_test = cifar_non_iid_single_class(dataset_train, dataset_test, args.num_clients, 10)  
        Net = Net_cifar        
    

    if args.experiment == 'performance-mnist':
        acc_final = [[] for i in range(3)]
        pro_final = [[] for i in range(3)]
        for m in range(3):
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
                    w_global = FedAvg(w_local)
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
                    pro_train.append(acc_client_num_95/args.num_clients*100)
                    print("Testing proportion: {:.1f}".format(acc_client_num_95/args.num_clients*100))          
            
            if args.method == 'CFMTL':
                groups = [[i for i in range(args.num_clients)]]
                loss_train = []
                pro_train = pro_final[m]
                acc_train = acc_final[m]
                w_global = Net().state_dict()
                w_groups = [Net().state_dict()]         
                for iter in range(args.ep):
                    loss_local = []
                    num_group = len(groups)
                    for group_id in range(num_group):
                        if iter == 0:
                            group = groups[group_id]
                        else:
                            group = groups[group_id]
                            num_clients = max(int(args.frac * len(group)), 1)
                            clients = np.random.choice(group, num_clients, replace=False)
                            group = clients
                        w_group = w_groups[group_id]
                        w_local = []
                        for id in group:
                            mem, w, loss = Local_Update(args, w_group, dataset_train, dict_train[id], iter)
                            w_local.append(w)
                            loss_local.append(loss)
                        if iter == 0:
                            groups, w_groups, rel = Cluster(group, w_local, args)
                        else:
                            w_groups[group_id] = FedAvg(w_local)               
                           
                    if len(groups) > 1 and args.prox is True:
                        w_groups = Prox(w_groups, args, rel)
                        
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
                    pro_train.append(acc_client_num_95/args.num_clients*100)       
                    print("Testing proportion: {:.1f}".format(acc_client_num_95/args.num_clients*100))                 
        record = []
        record.append(copy.deepcopy(acc_final))
        record.append(copy.deepcopy(pro_final))
       
    if args.experiment == 'performance-cifar':
        acc_final = [[] for i in range(3)]
        for m in range(3):
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
                    w_global = FedAvg(w_local)
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
            
            if args.method == 'CFMTL':
                groups = [[i for i in range(args.num_clients)]]
                loss_train = []
                acc_train = acc_final[m]
                w_global = Net().state_dict()
                w_groups = [Net().state_dict()]         
                for iter in range(args.ep):
                    loss_local = []
                    num_group = len(groups)
                    for group_id in range(num_group):
                        if iter == 0:
                            group = groups[group_id]
                        else:
                            group = groups[group_id]
                            num_clients = max(int(args.frac * len(group)), 1)
                            clients = np.random.choice(group, num_clients, replace=False)
                            group = clients
                        w_group = w_groups[group_id]
                        w_local = []
                        for id in group:
                            mem, w, loss = Local_Update(args, w_group, dataset_train, dict_train[id], iter)
                            w_local.append(w)
                            loss_local.append(loss)
                        if iter == 0:
                            groups, w_groups, rel = Cluster(group, w_local, args)
                        else:
                            w_groups[group_id] = FedAvg(w_local)               
                           
                    if len(groups) > 1 and args.prox is True:
                        w_groups = Prox(w_groups, args, rel)
                        
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
        record = []
        record.append(copy.deepcopy(acc_final))
        
    if args.experiment == 'communication':
        acc_final = [[] for i in range(2)]
        total_mem = [[] for i in range(2)]
        for m in range(2):
            if m == 0:
                args.method = 'CFMTL'
                args.prox = True
                args.if_clust = False
            if m == 1:
                args.method = 'CFMTL'
                args.prox = True
                args.if_clust = True        
            
            if args.method == 'CFMTL':
                groups = [[i for i in range(args.num_clients)]]
                loss_train = []
                acc_train = acc_final[m]
                mem_use = total_mem[m]
                w_global = Net().state_dict()
                w_groups = [Net().state_dict()]         
                for iter in range(args.ep):
                    loss_local = []
                    num_group = len(groups)
                    mem_sum = 0
                    for group_id in range(num_group):
                        if iter == 0:
                            group = groups[group_id]
                        else:
                            group = groups[group_id]
                            num_clients = max(int(args.frac * len(group)), 1)
                            clients = np.random.choice(group, num_clients, replace=False)
                            group = clients
                        w_group = w_groups[group_id]
                        w_local = []
                        for id in group:
                            mem, w, loss = Local_Update(args, w_group, dataset_train, dict_train[id], iter)
                            mem_sum += mem
                            w_local.append(w)
                            loss_local.append(loss)
                        if iter == 0:
                            groups, w_groups, rel = Cluster(group, w_local, args)
                        else:
                            w_groups[group_id] = FedAvg(w_local)               
                           
                    if len(groups) > 1 and args.prox is True:
                        w_groups = Prox(w_groups, args, rel)
                        
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
                    if iter != 0:
                        mem_sum += mem_use[-1]
                    mem_use.append(mem_sum)
                    print("Total amount of communication: {:.2f} MB".format(mem_use[-1]))
        record = []
        record.append(copy.deepcopy(total_mem))
        record.append(copy.deepcopy(acc_final))

    if args.experiment == 'hyperparameters':
        acc_final = [[] for i in range(3)]
        pro_final = [[] for i in range(3)]
        for m in range(3):
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
                    w_global = FedAvg(w_local)
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
                    pro_train.append(acc_client_num_95/args.num_clients*100)
                    print("Testing proportion: {:.1f}".format(acc_client_num_95/args.num_clients*100))          
            
            if args.method == 'CFMTL':
                groups = [[i for i in range(args.num_clients)]]
                loss_train = []
                pro_train = pro_final[m]
                acc_train = acc_final[m]
                w_global = Net().state_dict()
                w_groups = [Net().state_dict()]         
                for iter in range(args.ep):
                    loss_local = []
                    num_group = len(groups)
                    for group_id in range(num_group):
                        if iter == 0:
                            group = groups[group_id]
                        else:
                            group = groups[group_id]
                            num_clients = max(int(args.frac * len(group)), 1)
                            clients = np.random.choice(group, num_clients, replace=False)
                            group = clients
                        w_group = w_groups[group_id]
                        w_local = []
                        for id in group:
                            mem, w, loss = Local_Update(args, w_group, dataset_train, dict_train[id], iter)
                            w_local.append(w)
                            loss_local.append(loss)
                        if iter == 0:
                            groups, w_groups, rel = Cluster(group, w_local, args)
                        else:
                            w_groups[group_id] = FedAvg(w_local)               
                           
                    if len(groups) > 1 and args.prox is True:
                        w_groups = Prox(w_groups, args, rel)
                        
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
                    pro_train.append(acc_client_num_95/args.num_clients*100)       
                    print("Testing proportion: {:.1f}".format(acc_client_num_95/args.num_clients*100))                 
        record = []
        record.append(copy.deepcopy(acc_final))
        record.append(copy.deepcopy(pro_final))
        
    if args.experiment == 'metric':
        acc_final = [[] for i in range(4)]
        pro_final = [[] for i in range(4)]
        for m in range(4):
            if m == 0:
                args.method = 'CFMTL'
                args.prox = True
                args.if_clust = True
                args.dist = 'L2'
            if m == 1:
                args.method = 'CFMTL'
                args.prox = True
                args.if_clust = True
                args.dist = 'Equal'
            if m == 2:
                args.method = 'CFMTL'
                args.prox = True
                args.if_clust = True
                args.dist = 'L1'
            if m == 3:
                args.method = 'CFMTL'
                args.prox = True
                args.if_clust = True
                args.dist = 'Cos'
            
            if args.method == 'CFMTL':
                groups = [[i for i in range(args.num_clients)]]
                loss_train = []
                pro_train = pro_final[m]
                acc_train = acc_final[m]
                w_global = Net().state_dict()
                w_groups = [Net().state_dict()]         
                for iter in range(args.ep):
                    loss_local = []
                    num_group = len(groups)
                    for group_id in range(num_group):
                        if iter == 0:
                            group = groups[group_id]
                        else:
                            group = groups[group_id]
                            num_clients = max(int(args.frac * len(group)), 1)
                            clients = np.random.choice(group, num_clients, replace=False)
                            group = clients
                        w_group = w_groups[group_id]
                        w_local = []
                        for id in group:
                            mem, w, loss = Local_Update(args, w_group, dataset_train, dict_train[id], iter)
                            w_local.append(w)
                            loss_local.append(loss)
                        if iter == 0:
                            groups, w_groups, rel = Cluster(group, w_local, args)
                        else:
                            w_groups[group_id] = FedAvg(w_local)               
                           
                    if len(groups) > 1 and args.prox is True:
                        w_groups = Prox(w_groups, args, rel)
                        
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
                    pro_train.append(acc_client_num_95/args.num_clients*100)       
                    print("Testing proportion: {:.1f}".format(acc_client_num_95/args.num_clients*100))                 
        record = []
        record.append(copy.deepcopy(acc_final))
        record.append(copy.deepcopy(pro_final))
    
    record = np.array(record)
    np.save('{}.npy'.format(args.filename),record)
    

        

