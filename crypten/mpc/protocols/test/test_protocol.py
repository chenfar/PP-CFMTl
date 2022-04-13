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


# def eqz_2PC(x):
#     """Returns self == 0"""
#     """
#         这个是用于crypten两方时，比较x是否等于0，会调用BinarySharedTensor的比较方法。
#     """
#     # Create BinarySharedTensors from shares
#     x0 = BinarySharedTensor(x.share, src=0)
#     x1 = BinarySharedTensor(-x.share, src=1)
#
#     # Perform equality testing using binary shares
#     x0 = x0.eq(x1)
#     x0.encoder = x.encoder
#
#     # Convert to Arithmetic sharing
#     result = convert(x0, Ptype.arithmetic, bits=1)
#     result.encoder._scale = 1
#
#     return result


def SelectShare(alpha: ArithmeticSharedTensor, x: ArithmeticSharedTensor, y: ArithmeticSharedTensor):
    zero_u = ArithmeticSharedTensor(torch.zeros(alpha.share.shape))
    w = y - x
    c = alpha * w
    z = x + c + zero_u
    return z


torch.set_printoptions(sci_mode=False)


# def Compare(x: ArithmeticSharedTensor, y: ArithmeticSharedTensor) -> ArithmeticSharedTensor:
#     alpha = generate_random_ring_element(x.share.shape, 2 ** 1)
#     alpha = ArithmeticSharedTensor.from_shares(alpha)
#     ones = ArithmeticSharedTensor(torch.ones(x.share.shape))
#     r = generate_random_positive_ring_element(x.share.shape)
#     mid = (alpha * (x - y) + (-alpha + 1) * (y + ones - x)) * r
#     alpha_ = TrustedThirdParty.testnn_Check(mid)
#     return alpha + alpha_ - 2 * alpha * alpha_


class TestProtocol(BaseProtocol):
    """
    paper实验协议
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
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        alpha = torch.randint(0, 2, x.share.shape)
        # info(alpha)
        alpha = ArithmeticSharedTensor(alpha)
        # one = ArithmeticSharedTensor(torch.ones(x.share.shape))
        # ones = ArithmeticSharedTensor(torch.ones(x.share.shape)*(10**-14))
        # two=ArithmeticSharedTensor(torch.ones(x.share.shape)*2)
        r = ArithmeticSharedTensor.from_shares(
            generate_random_positive_ring_element(x.share.shape, 2 ** 32))
        # shift = torch.iinfo(torch.long).bits - 1
        mid = (alpha * x + (-alpha + 1) * (- x + (2 ** -16))) * r
        alpha_ = TrustedThirdParty.testnn_Check(mid)
        # two=ones*2
        result = alpha + alpha_ - alpha * alpha_ * 2
        # info(result.reveal())
        return result

    # @staticmethod
    # def ge(x, y):
    #     """Returns x >= y"""
    #     return Compare(x, y)
    #
    # @staticmethod
    # def gt(x, y):
    #     """Returns x > y"""
    #     return 1-Compare(y , x)
    #
    # @staticmethod
    # def le(x, y):
    #     """Returns x <= y"""
    #     return Compare(y, x)
    #
    # @staticmethod
    # def lt(x, y):
    #     """Returns x < y"""
    #     return 1-Compare(x , y)
    #
    # @staticmethod
    # def eq(x, y):
    #     """Returns x == y"""
    #     # if comm.get().get_world_size() == 2:
    #     #     return eqz_2PC(x - y)
    #
    #     return 1 - TestProtocol.ne(x, y)
    #
    # @staticmethod
    # def ne(x, y):
    #     """Returns x != y"""
    #     # if comm.get().get_world_size() == 2:
    #     #     return 1 - TestProtocol.eq(x, y)
    #     alpha=Compare(x,y)
    #     alpha_=Compare(y,x)
    #     return alpha + alpha_-2*alpha*alpha_
    @staticmethod
    def ge(x, y):
        """Returns x >= y"""
        return TestProtocol.ltz(-x + y)

    @staticmethod
    def gt(x, y):
        """Returns x > y"""
        return 1 - TestProtocol.le(x, y)

    @staticmethod
    def le(x, y):
        """Returns x <= y"""
        return TestProtocol.ltz(x - y)

    @staticmethod
    def lt(x, y):
        """Returns x < y"""
        return 1 - TestProtocol.ge(x, y)

    @staticmethod
    def eq(x, y):
        """Returns x == y"""
        a = TestProtocol.le(x, y)
        b = TestProtocol.ge(x, y)
        return 1 - (a + b - a * b * 2)

        # return 1 - TestProtocol.ne(x, y)

    @staticmethod
    def ne(x, y):
        """Returns x != y"""
        if comm.get().get_world_size() == 2:
            return 1 - TestProtocol.eq(x, y)

        difference = x - y
        difference.share = torch_stack([difference.share, -difference.share])
        return TestProtocol.ltz(difference).sum(0)
