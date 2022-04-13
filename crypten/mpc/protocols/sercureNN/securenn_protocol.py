import random

import torch


from .. import CrypTenProtocol
from ..base_protocol import BaseProtocol
from ...ptype import ptype as Ptype
from ...primitives import BinarySharedTensor
from ...primitives import ArithmeticSharedTensor
from ...primitives.beaver import beaver_protocol
from ...primitives.converters import convert
from ...provider.ttp_provider import TrustedThirdParty
from crypten import mpc, cryptensor
import crypten.communicator as comm
from ....common.rng import generate_random_ring_element, generate_random_positive_ring_element
from ....common.util import torch_stack
import torch.distributed as dist
import numpy as np


def info(*data):
    if dist.get_rank() == 0:
        print(*data)

"""
 选择分享函数，通过给定的alpha对x,y进行选择，根据alpha选择最后得到x还是y
"""
def SelectShare(alpha: ArithmeticSharedTensor, x: ArithmeticSharedTensor, y: ArithmeticSharedTensor):
    zero_u = ArithmeticSharedTensor(torch.zeros(alpha.share.shape))
    w = y - x
    c = alpha * w
    z = x + c + zero_u
    return z
# 选择函数

torch.set_printoptions(sci_mode=False)

"""
本SecureNN取消了模在协议中的作用 即不进行模计算 进行取模会使计算产生误差影响结果
比较协议的输出结果是bate xor (x>r) 协议实现的原理与falcon原理几乎相同。
"""


def PrivateCompare(x: ArithmeticSharedTensor, r: torch, beta: ArithmeticSharedTensor, l) -> ArithmeticSharedTensor:
    # 产生r与t的二进制
    t = r + 1
    r_bin = torch.rand(r.shape)
    for i in range(l):
        if i == 0:
            r_bin = r % 2
            r //= 2
            r_bin = r_bin.unsqueeze(-1)
        else:
            mid = r % 2
            r //= 2
            mid = mid.unsqueeze(-1)
            r_bin = torch.cat((mid, r_bin), dim=-1)

    t_bin = torch.rand(r.shape)
    for i in range(l):
        if i == 0:
            t_bin = t % 2
            t //= 2
            t_bin = t_bin.unsqueeze(-1)
        else:
            mid = t % 2
            t //= 2
            mid = mid.unsqueeze(-1)
            t_bin = torch.cat((mid, t_bin), dim=-1)
    # 计算w与c
    w_enc = ArithmeticSharedTensor(torch.zeros(x.share.shape, dtype=torch.int32))
    w_sum = ArithmeticSharedTensor(torch.zeros(x.share.shape[:-1], dtype=torch.int32))
    c_enc = ArithmeticSharedTensor(torch.zeros(x.share.shape, dtype=torch.int32))

    for i in range(0, l):
        if beta.get_plain_text().item() == 0:
            w_enc[..., i] = x[..., i] + r_bin[..., i] - x[..., i] * r_bin[..., i] * 2
            c_enc[..., i] = r_bin[..., i] - x[..., i]  + w_sum + 1
            w_sum = w_sum + w_enc[..., i]
        else:
            w_enc[..., i] = x[..., i] + t_bin[..., i] - x[..., i] * t_bin[..., i] * 2
            c_enc[..., i] = x[..., i] - t_bin[..., i] + w_sum + 1
            w_sum = w_sum + w_enc[..., i]

    # 传给P2进行0存在判断  这里P2用可信第三方
    perm=torch.randperm(l)
    enc_perm=ArithmeticSharedTensor(perm)
    d_enc = ArithmeticSharedTensor(torch.zeros(c_enc.share.shape, dtype=torch.int32))
    for i in range(l):
        d_enc[..., i] = c_enc[..., enc_perm.get_plain_text()[i].int()]
    beta_ = TrustedThirdParty.Securenn_PC_Reconst(d_enc)

    return beta_

# """这个函数由于模运算无法使用 该函数基本没有作用了"""
# def ShareConvert(a: ArithmeticSharedTensor, l) -> ArithmeticSharedTensor:
#     r_ = generate_random_positive_ring_element(a.share.shape, 2 ** l)
#     r = ArithmeticSharedTensor.from_shares(r_)
#     r_ = r.get_plain_text()
#     eta__ = torch.tensor(random.randint(0, 1))
#     eta__ = ArithmeticSharedTensor(eta__)
#     alpha = CrypTenProtocol.ge(r, 2 ** l)
#     zero = ArithmeticSharedTensor(torch.zeros(a.share.shape))
#     tilde_a = a + r
#     beta = ArithmeticSharedTensor.from_shares(((a.share + r.share) >= 2 ** l).int())
#     enc_x_bit, delta = TrustedThirdParty.Securenn_SC_Reconst(tilde_a, l)
#     eta_ = PrivateCompare(enc_x_bit, r_ - 1, eta__, l)
#     eta = eta_ + eta__ - 2 * eta_ * eta__
#     theta = beta - alpha + delta + eta - 1
#     y = a - theta + zero
#     return y


# '''以下函数未经过测试 需要等待架构调整'''
def ComputeMSB(a: ArithmeticSharedTensor, l) -> ArithmeticSharedTensor:
    return SecureNNProtocol.ltz(a)


def DReLU(a: ArithmeticSharedTensor, l) -> ArithmeticSharedTensor:
    return 1 - ComputeMSB(a, l)


def ReLU(a: ArithmeticSharedTensor, l) -> ArithmeticSharedTensor:
    return a * DReLU(a, l)


def Division(x: ArithmeticSharedTensor, y: ArithmeticSharedTensor, l):
    zero_w = [ArithmeticSharedTensor(torch.zeros(x.share.shape)) for i in range(l)]
    zero_s = ArithmeticSharedTensor(torch.zeros(x.share.shape))
    u = ArithmeticSharedTensor(torch.zeros(x.share.shape))
    q = ArithmeticSharedTensor(torch.zeros(x.share.shape))
    for i in range(l):
        z = x - u - 2 ** (l - 1 - i) * y + zero_w[i]
        beta = SecureNNProtocol.ge(z, 0)
        # z.share=z.share%(2**l)
        # beta = DReLU(z, l)
        # print(beta.get_plain_text())
        v = beta * 2 ** (l - 1 - i) * y
        q += 2 ** (l - 1 - i) * beta
        u = u + v
    return q + zero_s


# def Maxpool(x: ArithmeticSharedTensor, l, n):
#     zero_u = ArithmeticSharedTensor(torch.zeros(x.share.shape[:-1]))
#     zero_v = ArithmeticSharedTensor(torch.zeros(x.share.shape[:-1]))
#     max_ans = x[..., 0]
#     max_index = ArithmeticSharedTensor(0)
#     for i in range(n):
#         w = x[..., i] - max_ans
#         beta = SecureNNProtocol.ge(w, 0)
#         max_ans = SelectShare(beta, max_ans, x[..., i])
#         max_index = SelectShare(beta, max_index, ArithmeticSharedTensor(i))
#     max_ans = max_ans + zero_u
#     max_index = max_index + zero_v
#     return max_ans, max_index


# def DMaxpool(x: ArithmeticSharedTensor, l, n):
#     max_ans, max_index = Maxpool(x, l, n)
#     info(max_index.get_plain_text())
#     r_ = generate_random_positive_ring_element(max_index.share.shape, 2 ** l)
#     r = ArithmeticSharedTensor.from_shares(r_, precision=0)
#     max_index_r = max_index + r
#     E = TrustedThirdParty.Securenn_DMaxpool_Reconst(max_index_r, l, n)
#     g = r.get_plain_text() % n
#     index = torch.empty((1), dtype=torch.int32)
#     for i in range(n):
#         if i == 0:
#             index = torch.randint(i, i + 1, g.size()).unsqueeze(-1)
#         else:
#             mid = torch.randint(i, i + 1, g.size()).unsqueeze(-1)
#             index = torch.cat((index, mid), dim=-1)
#     g = g.unsqueeze(-1).expand_as(index)
#     indexx = ((index + g) % n).int()
#     D = torch.gather(E.share, len(indexx.size()) - 1, indexx)
#     D = ArithmeticSharedTensor.from_shares(D)
#     return D


class SecureNNProtocol(BaseProtocol):
    """
    增加第三方十进制转二进制操作 使用privatecompare协议比较 MSB协议无法适配该框架
    """

    @staticmethod
    def mul(x, y):
        return beaver_protocol("mul", x, y)

    @staticmethod
    def matmul(x, y):
        return beaver_protocol("matmul", x, y)

    @staticmethod
    def conv1d(x, y, **kwargs):
        return beaver_protocol("conv1d", x, y, **kwargs)

    @staticmethod
    def conv2d(x, y, **kwargs):
        return beaver_protocol("conv2d", x, y, **kwargs)

    @staticmethod
    def conv_transpose1d(x, y, **kwargs):
        return beaver_protocol("conv_transpose1d", x, y, **kwargs)

    @staticmethod
    def conv_transpose2d(x, y, **kwargs):
        return beaver_protocol("conv_transpose2d", x, y, **kwargs)

    @staticmethod
    def select_share(a, x, y):
        """Returns a==0?x:y"""
        return SelectShare(a, x, y)

    @staticmethod
    def ltz(x):
        """ 输出的结果是 x>0,输出1   x<=0,输出0"""
        l = 64
        rr, enc_r_bit = TrustedThirdParty.Get_Rand_Bit(x.size())
        x_ = (x + rr)
        beta = ArithmeticSharedTensor(random.randint(0, 1))
        beta_ = PrivateCompare(enc_r_bit, x_.reveal(), beta, l)
        gamma = beta + beta_ - beta * beta_ * 2
        return gamma
    @staticmethod
    def ge(x, y):
        """Returns x >= y"""
        return 1 - SecureNNProtocol.lt(x, y)

    @staticmethod
    def gt(x, y):
        """Returns x > y"""
        return SecureNNProtocol.ltz(-x + y)

    @staticmethod
    def le(x, y):
        """Returns x <= y"""
        return 1 - SecureNNProtocol.gt(x, y)

    @staticmethod
    def lt(x, y):
        """Returns x < y"""
        return SecureNNProtocol.ltz(x - y)

    @staticmethod
    def eq(x, y):
        """Returns x == y"""
        a = SecureNNProtocol.le(x, y)
        b = SecureNNProtocol.ge(x, y)
        return 1 - (a + b - a * b * 2)

    @staticmethod
    def ne(x, y):
        """Returns x != y"""
        if comm.get().get_world_size() == 2:
            return 1 - SecureNNProtocol.eq(x, y)

        difference = x - y
        difference.share = torch_stack([difference.share, -difference.share])
        return SecureNNProtocol.ltz(difference).sum(0)
