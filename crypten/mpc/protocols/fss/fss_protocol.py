"""Function Secret Sharing Protocol.
ARIANN: Low-Interaction Privacy-Preserving Deep Learning via Function Secret Sharing
arXiv:2006.04593 [cs.LG]
"""
import multiprocessing

import numpy as np
import sycret
import torch

import crypten
from crypten.mpc.primitives import ArithmeticSharedTensor
from crypten.mpc.primitives.beaver import beaver_protocol
from crypten.mpc.protocols.base_protocol import BaseProtocol
import torch.distributed as dist

n = 32  # bit precision
N_CORES = multiprocessing.cpu_count()
dpf = sycret.EqFactory(n_threads=N_CORES)
dif = sycret.LeFactory(n_threads=N_CORES)


def fss_op(x: ArithmeticSharedTensor, op="eq") -> ArithmeticSharedTensor:
    """Define the workflow for a binary operation using Function Secret Sharing.

        Currently supported operand are = & <=, respectively corresponding to

        op = "eq"    if x == 0, return 1, else return 0
        op = "comp"  if x <= 0, return 1, else return 0

        Args:
            x (ArithmeticSharedTensor):  private value.
            op: Type of operation to perform, should be 'eq' or 'comp'. Defaults to eq.

        Returns:
            ArithmeticSharedTensor: Shares of the comparison.
    """
    # TODO: if input x is cuda tensor, should do something.
    device = x.device
    if str(device) != "cpu":
        x = x.clone().cpu()
    rank = dist.get_rank()
    origin_shape = x.share.shape
    n_values = origin_shape.numel()

    # get keys from provider
    # TODO: now the keys only support tfp, need to add ttp support
    provider = crypten.mpc.get_default_provider()
    keys = provider.generate_fss_keys(
        rank=rank, n_values=n_values, op=op).numpy()
    # mask x with keys
    alpha = np.frombuffer(np.ascontiguousarray(
        keys[:, 0:n // 8]), dtype=np.uint32)

    x._tensor += torch.tensor(alpha.astype(np.int64)).reshape(origin_shape)

    # public x_masked
    x_masked = x.reveal()

    # evaluate
    x_masked = x_masked.numpy().reshape(-1)

    if op == "eq":
        flat_result = dpf.eval(rank, x_masked, keys)
    elif op == "comp":
        flat_result = dif.eval(rank, x_masked, keys)
    else:
        raise ValueError(f"{op} is an FSS unsupported operation.")

    # build result as ArithmeticSharedTensor
    # numpy => tensor => ArithmeticSharedTensor
    result_share = flat_result.astype(np.int32).astype(
        np.int64).reshape(origin_shape)

    result_tensor = torch.tensor(result_share, dtype=torch.int64)

    result = ArithmeticSharedTensor.from_shares(
        result_tensor, precision=0, device=device)
    return result


class FSSProtocol(BaseProtocol):
    """
        此协议只支持两方计算，并且依赖可信第三方（可信第一方的话也不是不行。。。）
        乘法还是使用三元组
        比较使用FSS，函数秘密共享，function secret sharing
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
        """Retuens x < 0"""
        return 1 - fss_op(-x, op="comp")

    @staticmethod
    def lt(x, y):
        """Returns x < y"""
        return 1 - FSSProtocol.ge(x, y)

    @staticmethod
    def le(x, y):
        """Returns x <= y"""
        return fss_op(x - y, op="comp")

    @staticmethod
    def gt(x, y):
        """Returns x > y"""
        return 1 - FSSProtocol.le(x, y)

    @staticmethod
    def ge(x, y):
        """Returns x >= y"""
        return fss_op(-x + y, op="comp")

    @staticmethod
    def eq(x, y):
        """Returns x == y"""
        return fss_op(x - y)

    @staticmethod
    def ne(x, y):
        """Returns x != y"""
        return 1 - fss_op(x - y)
