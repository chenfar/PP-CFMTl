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
falcon 主要实现的就是比较函数，比较函数的输入是待比较的x和r,输出的结果是 beta xor (x>=r). 实现原理则是如果出现二进制表示下，某一位不相同，但是该位前面
的所有的位都相同(这种情况就可以说明两者的大小关系了),那么可以求出一个c_enc是等于0的。最后将所有的c_enc相乘，如果存在一个0那么最后的结果一定是0。所以
如果最后返回的是0，再根据beta的值就可以获得比较的结果。
"""

def PrivateCompare(x: ArithmeticSharedTensor, r: torch, beta: ArithmeticSharedTensor, l) -> ArithmeticSharedTensor:
    # r的二进制转化
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
    w_enc = ArithmeticSharedTensor(torch.zeros(x.share.shape, dtype=torch.int32))
    w_sum = ArithmeticSharedTensor(torch.zeros(x.share.shape[:-1], dtype=torch.int32))
    c_enc = ArithmeticSharedTensor(torch.zeros(x.share.shape, dtype=torch.int32))
    for i in range(0, l):
        w_enc[..., i] = x[..., i] + r_bin[..., i] - x[..., i] * r_bin[..., i]*2
        c_enc[..., i] = (beta * 2 -1) * (r_bin[..., i] - x[..., i])  + w_sum + 1
        w_sum = w_sum + w_enc[..., i]
    m=generate_random_positive_ring_element(c_enc[..., 0].share.shape,2**64)
    for i in range(l):
        m = c_enc[..., i]*m

    """原论文是直接揭露d 这里为了保持与securenn相同架构 同样用第三方验证d （完全按照论文流程也是可以的 只需要把函数提取出来方下面就行） 这里没有安全性影响"""

    beta_ = TrustedThirdParty.Falcon_PC_Reconst(m)
    gamma = beta + beta_ - beta * beta_ * 2
    return gamma

class FalconProtocol(BaseProtocol):
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
    def ltz(x):
        """ ltz函数实现的的是如果x<=0,输出1   x>0,输出0 """
        l = 64
        rr, enc_r_bit = TrustedThirdParty.Get_Rand_Bit(x.size())
        x_ = (x + rr)
        beta = ArithmeticSharedTensor(torch.randint(0, 1, x.size()))
        gamma = PrivateCompare(enc_r_bit, x_.reveal(), beta, l)
        return gamma

    @staticmethod
    def ge(x, y):
        """Returns x >= y"""
        return FalconProtocol.ltz(-x + y)

    @staticmethod
    def gt(x, y):
        """Returns x > y"""
        return 1 - FalconProtocol.le(x, y)

    @staticmethod
    def le(x, y):
        """Returns x <= y"""
        return FalconProtocol.ltz(x - y)

    @staticmethod
    def lt(x, y):
        """Returns x < y"""
        return 1 - FalconProtocol.ge(x, y)

    @staticmethod
    def eq(x, y):
        """Returns x == y"""
        # if comm.get().get_world_size() == 2:
        #     return eqz_2PC(x - y)

        # return 1 - SecureNNProtocol.ne(x, y)
        a = FalconProtocol.le(x, y)
        b = FalconProtocol.ge(x, y)
        return 1 - (a + b - a * b * 2)

    @staticmethod
    def ne(x, y):
        """Returns x != y"""
        if comm.get().get_world_size() == 2:
            return 1 - FalconProtocol.eq(x, y)

        difference = x - y
        difference.share = torch_stack([difference.share, -difference.share])
        return FalconProtocol.ltz(difference).sum(0)
