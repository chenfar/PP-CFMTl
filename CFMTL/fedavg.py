import torch

def FedAvg(w):
    w_avg = w[0]
    for i in w_avg.keys():
        for j in range(1, len(w)):
            w_avg[i] += w[j][i]
        w_avg[i] = torch.div(w_avg[i], len(w))
    return w_avg
