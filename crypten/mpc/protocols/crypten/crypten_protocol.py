import torch

from ..base_protocol import BaseProtocol
from ...ptype import ptype as Ptype
from ...primitives import BinarySharedTensor
from ...primitives.beaver import beaver_protocol
from ...primitives.converters import convert
import crypten.communicator as comm
from ....common.util import torch_stack


def eqz_2PC(x):
    """Returns self == 0"""
    """
        这个是用于crypten两方时，比较x是否等于0，会调用BinarySharedTensor的比较方法。
    """
    # Create BinarySharedTensors from shares
    x0 = BinarySharedTensor(x.share, src=0)
    x1 = BinarySharedTensor(-x.share, src=1)

    # Perform equality testing using binary shares
    x0 = x0.eq(x1)
    x0.encoder = x.encoder

    # Convert to Arithmetic sharing
    result = convert(x0, Ptype.arithmetic, bits=1)
    result.encoder._scale = 1

    return result


class CrypTenProtocol(BaseProtocol):
    """
    CrypTen的MPC协议封装类，乘法基于beaver_protocol，比较基于符号位 或者 BinarySharedTensor（两方）
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
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        shift = torch.iinfo(torch.long).bits - 1
        precision = 0 if x.encoder.scale == 1 else None

        result = convert(x, Ptype.binary)
        result.share >>= shift

        result = convert(result, Ptype.arithmetic, precision=precision, bits=1)
        result.encoder._scale = 1
        return result

    @staticmethod
    def ge(x, y):
        """Returns x >= y"""
        return 1 - CrypTenProtocol.lt(x, y)

    @staticmethod
    def gt(x, y):
        """Returns x > y"""
        return CrypTenProtocol.ltz(-x + y)

    @staticmethod
    def le(x, y):
        """Returns x <= y"""
        return 1 - CrypTenProtocol.gt(x, y)

    @staticmethod
    def lt(x, y):
        """Returns x < y"""
        return CrypTenProtocol.ltz(x - y)

    @staticmethod
    def eq(x, y):
        """Returns x == y"""
        if comm.get().get_world_size() == 2:
            return eqz_2PC(x - y)

        return 1 - CrypTenProtocol.ne(x, y)

    @staticmethod
    def ne(x, y):
        """Returns x != y"""
        if comm.get().get_world_size() == 2:
            return 1 - CrypTenProtocol.eq(x, y)

        difference = x - y
        difference.share = torch_stack([difference.share, -difference.share])
        return CrypTenProtocol.ltz(difference).sum(0)
