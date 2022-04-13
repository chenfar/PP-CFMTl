import torch

from crypten import mpc
from ..base_protocol import BaseProtocol
from ...ptype import ptype as Ptype
from ...primitives import BinarySharedTensor, ArithmeticSharedTensor
from ...primitives.beaver import beaver_protocol
from ...primitives.converters import convert
import crypten.communicator as comm
from ....common.util import torch_stack
import random
from crypten.common.rng import generate_random_ring_element, generate_random_positive_ring_element
from ...primitives.ot import baseOT
import torch.distributed as dist
from ...provider import TrustedThirdParty


def BitLT(a: torch, enc_b_bin: ArithmeticSharedTensor, l: int):
    # 生成a的二进制
    a_bin = 0
    for i in range(l):
        if i == 0:
            a_bin = a % 2
            a //= 2
            a_bin = a_bin.unsqueeze(-1)
        else:
            mid = a % 2
            a //= 2
            mid = mid.unsqueeze(-1)
            a_bin = torch.cat((mid, a_bin), dim=-1)
    # d=a xor b
    d = ArithmeticSharedTensor(torch.zeros(enc_b_bin.share.shape))
    for i in range(l):
        d[..., i] = -enc_b_bin[..., i] * a_bin[..., i] * \
            2 + a_bin[..., i] + enc_b_bin[..., i]
    # 连乘得到p
    enc_p_bin = ArithmeticSharedTensor(torch.zeros(enc_b_bin.share.shape))
    for i in range(l):
        if i == 0:
            enc_p_bin[..., i] = d[..., i] + 1
        else:
            enc_p_bin[..., i] = enc_p_bin[..., i-1] * (d[..., i] + 1)
    # 得到sk
    enc_s_bin = ArithmeticSharedTensor(torch.zeros(enc_b_bin.share.shape))
    for i in range(l):
        if i == 0:
            enc_s_bin[..., i] = enc_p_bin[..., i] - 1
        else:
            enc_s_bin[..., i] = enc_p_bin[..., i] - enc_p_bin[..., i-1]
    # 计算s
    enc_s = ArithmeticSharedTensor(torch.zeros(a.shape))
    for i in range(l):
        enc_s += enc_s_bin[..., i]*(-a_bin[..., i]+1)
    # 这里避开s对2取模 （精度问题） 改为先生成一个随机数ran及其二进制 揭露s+ran的奇偶性 与ran二进制的末位异或得到s的奇偶性 安全性不受影响
    enc_ran_bin = ArithmeticSharedTensor(torch.zeros(enc_b_bin.share.shape))
    enc_ran = ArithmeticSharedTensor(torch.zeros(a.shape))
    for i in range(20):
        r = torch.randint(0, 2, a.shape)
        enc_ran_bin[..., i] = ArithmeticSharedTensor(r)
        enc_ran += 2**i*r
    c = (enc_s+enc_ran).get_plain_text()
    c_0 = c % 2
    result = c_0+enc_ran_bin[..., 0]-enc_ran_bin[..., 0]*c_0*2
    return result

#以下函数均为测试函数，不同进程执行不同的函数操作,可以执行正确的生成三元组的操作以及验证数字的正确性
def Gen_rand_key(data_len):
    key_0 = []
    key_1 = []
    for i in range(data_len):
        key_0.append(str(random.randint(0, 1)))
        key_1.append(str(random.randint(0, 1)))
    return key_0, key_1


def AtoB(num, date_len):
    ans = [0 for i in range(date_len)]
    len = 0
    while num > 0:
        ans[len] = num % 2
        num //= 2
        len += 1
    return ans


def BtoA(num, date_len):
    ans = 0
    for i in range(date_len):
        ans += (num[i] * (2**i))
    return ans


def COPE_SERVER(a_tensor, data_len):
    process_randk = dist.get_rank()
    obvil_tran = baseOT.BaseOT(process_randk ^ 1)
    key_0, key_1 = Gen_rand_key(data_len)
    obvil_tran.send(key_0, key_1)
    num_count = 0
    t_0 = [0 for i in range(data_len)]
    t_1 = [0 for i in range(data_len)]
    u = [0 for i in range(data_len)]
    for i in range(data_len):
        random.seed(int(key_0[i]) + num_count)
        t_0[i] = random.randint(0, (2 ** 12))
        random.seed(int(key_1[i]) + num_count)
        t_1[i] = random.randint(0, (2 ** 12))
        u[i] = t_0[i] - t_1[i] + a_tensor
        comm.get().send_obj(u[i], (process_randk ^ 1))
        num_count += 1
    t = ((BtoA(t_0, data_len)) % (2 ** 60)) * -1
    ans = generate_random_positive_ring_element((1, 1), ring_size=(2 ** 5))
    ans[0][0] = t
    # Ans_Muti = ArithmeticSharedTensor.from_shares(ans, precision=0)
    return ans


def COPE_CLIENT(b_tensor, data_len):
    process_randk = dist.get_rank()
    obvil_tran = baseOT.BaseOT(process_randk ^ 1)
    b_binary = AtoB(b_tensor, data_len)
    key_x = obvil_tran.receive(b_binary)
    num_count = 0
    t_delt = [0 for i in range(data_len)]
    u = [0 for i in range(data_len)]
    q = [0 for i in range(data_len)]
    for i in range(data_len):
        random.seed(int(key_x[i]) + num_count)
        t_delt[i] = random.randint(0, (2 ** 12))
        u[i] = comm.get().recv_obj(process_randk ^ 1)
        q[i] = b_binary[i] * u[i] + t_delt[i]
        num_count += 1
    q = (BtoA(q, data_len)) % (2 ** 60)
    ans = generate_random_positive_ring_element((1, 1), ring_size=(2 ** 5))
    ans[0][0] = q
    # Ans_Muti = ArithmeticSharedTensor.from_shares(ans, precision=0)
    return ans


def GEN_TRIPLES():
    data_len = 64
    a = random.randint(0, 2 ** 7)
    b = random.randint(0, 2 ** 7)
    process_rank = dist.get_rank()
    if process_rank == 0:
        c = COPE_SERVER(a, data_len)
    else:
        c = COPE_CLIENT(b, data_len)
    if process_rank == 1:
        c += COPE_SERVER(a, data_len)
    else:
        c += COPE_CLIENT(b, data_len)
    c = ArithmeticSharedTensor.from_shares(c, precision=0)
    ans = generate_random_positive_ring_element((1, 1), ring_size=(2 ** 5))
    ans[0][0] = a*b
    temp = ArithmeticSharedTensor.from_shares(ans, precision=0)
    c += temp
    a_ = generate_random_positive_ring_element((1, 1), ring_size=(2 ** 5))
    b_ = generate_random_positive_ring_element((1, 1), ring_size=(2 ** 5))
    a_[0][0] = a
    b_[0][0] = b
    a = ArithmeticSharedTensor.from_shares(a_, precision=0)
    b = ArithmeticSharedTensor.from_shares(b_, precision=0)
    return a, b, c


def VERIFY_CORRECT():
    data_len = 64
    process_rank = dist.get_rank()
    if process_rank == 0:
        # 这是一个明文值，不是分享值，验证这个值是否是对的，这个值是x,通过计算x*dely,看看这个结果是不是等于m来判断这个x是不是被篡改，其中的delt是始终不变的，但是谁都不知道
        wait_verfiy = random.randint(0, 2 ** 7)
        print(wait_verfiy, process_rank, "\n")
    delt = random.randint(0, 2 ** 7)  # 这个是分享值，每个服务器上都有一个
    print(delt, "delt", process_rank, "\n")
    if process_rank == 0:
        m = COPE_SERVER(wait_verfiy, data_len)
        ans = generate_random_positive_ring_element((1, 1), ring_size=(2 ** 5))
        ans[0][0] = wait_verfiy * delt
        # temp = ArithmeticSharedTensor.from_shares(ans, precision=0)
        m += ans
    else:
        m = COPE_CLIENT(delt, data_len)
    m = ArithmeticSharedTensor.from_shares(m, precision=0)
    return m


class SPDZProtocol(BaseProtocol):

    @staticmethod
    def mul(x, y):
        mpc.config.active_security = True
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
        l = 64
        rr, enc_r_bit = TrustedThirdParty.Get_Rand_Bit(x, l)
        x_ = (x + rr)

        return BitLT(x_.reveal(), enc_r_bit, l)

    @staticmethod
    def ge(x, y):
        """Returns x >= y"""
        return 1 - SPDZProtocol.lt(x, y)

    @staticmethod
    def gt(x, y):
        """Returns x > y"""
        return SPDZProtocol.ltz(-x + y)

    @staticmethod
    def le(x, y):
        """Returns x <= y"""
        return 1 - SPDZProtocol.gt(x, y)

    @staticmethod
    def lt(x, y):
        """Returns x < y"""
        return SPDZProtocol.ltz(x - y)

    @staticmethod
    def eq(x, y):
        """Returns x == y"""
        # if comm.get().get_world_size() == 2:
        #     return eqz_2PC(x - y)

        # return 1 - SecureNNProtocol.ne(x, y)
        a = SPDZProtocol.le(x, y)
        b = SPDZProtocol.ge(x, y)
        return 1 - (a + b - a * b * 2)

    @staticmethod
    def ne(x, y):
        """Returns x != y"""
        if comm.get().get_world_size() == 2:
            return 1 - SPDZProtocol.eq(x, y)

        difference = x - y
        difference.share = torch_stack([difference.share, -difference.share])
        return SPDZProtocol.ltz(difference).sum(0)

    @staticmethod
    def gen_triple():
        return GEN_TRIPLES()

    @staticmethod
    def verify_correct():
        return VERIFY_CORRECT()
