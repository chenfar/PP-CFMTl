import torch

# import crypten.nn
import crypten

crypten.init()
a = crypten.cryptensor(torch.randn(1, 1).cuda())
a.log_softmax(dim=1)
