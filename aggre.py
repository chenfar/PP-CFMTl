from collections import OrderedDict


def fed_avg(W, num_c=None):
    w_avg = W[0]
    if num_c is None:
        num_c = len(W)  # W中真实有的客户端数量
    for k in w_avg.keys():
        for c in range(1, len(W)):
            w_avg[k] += W[c][k]
        w_avg[k] /= num_c
    return w_avg


def cluster_avg(one_hot, W):
    num_g = one_hot.size()[1]
    num_c = len(W)
    mix_w_groups = [[] for _ in range(num_g)]
    for c in range(num_c):
        for g in range(num_g):
            mix_w = OrderedDict()  # create a new state_dict
            c_in_g_code = one_hot[c][g]
            for k in W[c].keys():
                mix_w[k] = W[c][k] * c_in_g_code
            mix_w_groups[g].append(mix_w)

    new_w_groups = []
    num_c_s = one_hot.sum(0)
    for g in range(num_g):
        new_w_groups.append(fed_avg(mix_w_groups[g], num_c=num_c_s[g]))

    return new_w_groups


def client_w(one_hot, w_groups):
    client_ws = []
    num_c = one_hot.size()[0]
    num_g = one_hot.size()[1]
    for c in range(num_c):
        mix_w = OrderedDict()
        for k in w_groups[0].keys():
            mix_w[k] = w_groups[0][k] * one_hot[c][0]
        for g in range(1, num_g):
            for k in mix_w.keys():
                mix_w[k].add_(w_groups[g][k] * one_hot[c][g])
        client_ws.append(mix_w)
    return client_ws
