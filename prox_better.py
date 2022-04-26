from model import *
import crypten


def L2(_old_params, _old_w, param, args, rel):
    w = []
    for p in param:
        w.append(p.flatten())
    _w = crypten.cat(w)

    x = _w - _old_w
    x = x.norm()
    loss = x.pow(2)

    for i in range(len(_old_params)):
        _param = _old_params[i]
        x = _w - _param
        x = x.norm()
        x = x.pow(2)
        x = x.mul(args.L)
        x = x.mul(rel[i])
        loss = loss.add(x)

    return loss


def Prox(w_groups, args, rel):
    w = L2_Prox(w_groups, args, rel)
    return w


def L2_Prox(w_groups, args, rel):
    if args.dataset == 'mnist':
        Net = Crypten_Net_mnist
    else:
        Net = Crypten_Net_cifar

    old_params = []
    for w in w_groups:
        tmp = []
        for k in w.keys():
            tmp.append(w[k].flatten())
        old_params.append(crypten.cat(tmp).cuda())

    w_new = []
    for i in range(len(w_groups)):
        w = w_groups[i]
        net = Net().encrypt()
        net.load_state_dict(w, strict=False)
        net.cuda()
        opt = crypten.optim.SGD(net.parameters(), lr=args.prox_lr, momentum=args.prox_momentum)

        _old_params = crypten.stack(old_params[:i] + old_params[i + 1:])
        rel_i = crypten.stack(rel[i])
        rel_i = rel_i.cuda()
        for iter in range(args.prox_local_ep):
            # info(_old_params.get_plain_text())
            loss = L2(_old_params, old_params[i], net.parameters(), args, rel_i)
            # if iter == 0:
            #     loss_start = copy.deepcopy(loss)
            # if iter == args.prox_local_ep - 1:
            #     loss_end = copy.deepcopy(loss)
            #     percent = (loss_end - loss_start).get_plain_text() / loss_start.get_plain_text() * 100
            #     print("Percent: {:.2f}%".format(percent))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        w_new.append(net.state_dict())

    return w_new
