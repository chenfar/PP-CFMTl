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


def eqz_2PC(x):
    """Returns self == 0"""
    """
        恶意协议 产生随机结果
    """
    result = torch.randint(0, 2, x.share.shape)
    return ArithmeticSharedTensor(result)


torch.set_printoptions(sci_mode=False)


class MaliciousProtocol(BaseProtocol):
    """
    恶意协议 产生随机结果
    """

    @staticmethod
    def mul(x, y):
        result = beaver_protocol("mul", x, y)
        return ArithmeticSharedTensor(torch.rand(result.share.shape))

    @staticmethod
    def matmul(x, y):
        result = beaver_protocol("matmul", x, y)
        return ArithmeticSharedTensor(torch.rand(result.share.shape))

    @staticmethod
    def conv1d(x, y, **kwargs):
        result = beaver_protocol("conv1d", x, y, **kwargs)
        return ArithmeticSharedTensor(torch.rand(result.share.shape))

    @staticmethod
    def conv2d(x, y, **kwargs):
        result = beaver_protocol("conv2d", x, y, **kwargs)
        return ArithmeticSharedTensor(torch.rand(result.share.shape))

    @staticmethod
    def conv_transpose1d(x, y, **kwargs):
        result = beaver_protocol("conv_transpose1d", x, y, **kwargs)
        return ArithmeticSharedTensor(torch.rand(result.share.shape))

    @staticmethod
    def conv_transpose2d(x, y, **kwargs):
        result = beaver_protocol("conv_transpose2d", x, y, **kwargs)
        return ArithmeticSharedTensor(torch.rand(result.share.shape))

    @staticmethod
    def ltz(x):
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        result = torch.randint(0, 2, x.share.shape)
        return ArithmeticSharedTensor(result)

    @staticmethod
    def ge(x, y):
        """Returns x >= y"""
        return 1 - MaliciousProtocol.lt(x, y)

    @staticmethod
    def gt(x, y):
        """Returns x > y"""
        return MaliciousProtocol.ltz(-x + y)

    @staticmethod
    def le(x, y):
        """Returns x <= y"""
        return 1 - MaliciousProtocol.gt(x, y)

    @staticmethod
    def lt(x, y):
        """Returns x < y"""
        return MaliciousProtocol.ltz(x - y)

    @staticmethod
    def eq(x, y):
        """Returns x == y"""
        if comm.get().get_world_size() == 2:
            return eqz_2PC(x - y)

        return 1 - MaliciousProtocol.ne(x, y)

    @staticmethod
    def ne(x, y):
        """Returns x != y"""
        if comm.get().get_world_size() == 2:
            return 1 - MaliciousProtocol.eq(x, y)

        difference = x - y
        difference.share = torch_stack([difference.share, -difference.share])
        return MaliciousProtocol.ltz(difference).sum(0)
