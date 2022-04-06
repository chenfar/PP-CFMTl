import argparse
from model import *
import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

class DatasetFolder(Dataset):
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        images, labels = self.dataset[self.ids[item]]
        return images, labels

def Local_Update(args, w, dataset = None, ids = None, round = 0):
    dataloader = DataLoader(DatasetFolder(dataset, ids), batch_size = args.num_batch, shuffle=True)
    
    if args.dataset == 'mnist':
        Net = Net_mnist
    if args.dataset == 'cifar':
        Net = Net_cifar
    
    net = Net()
    net.load_state_dict(w)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr/(1+round*args.decay), momentum=args.momentum)
    
    ep_loss = []
    for iter in range(args.local_ep):
        for data in dataloader:
            images, labels = data
            net.zero_grad()
            log_probs = net(images)
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())
    mem = sum([param.nelement()*param.element_size() for param in net.parameters()])
    mem *= 1e-6
    return mem, net.state_dict(), sum(ep_loss) / len(ep_loss)