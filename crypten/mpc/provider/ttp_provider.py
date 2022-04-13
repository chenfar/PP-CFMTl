#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import random

import sycret
import crypten
import crypten.communicator as comm
import torch
import torch.distributed as dist
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element, generate_random_positive_ring_element
from crypten.common.util import count_wraps
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor
from ..primitives.converters import convert
from ..ptype import ptype as Ptype
from ...encoder import FixedPointEncoder
import numpy as np

TTP_FUNCTIONS = ["additive", "square", "binary", "wraps", "B2A"]


def info(*data):
    if dist.get_rank() == 0:
        print(*data)


class TrustedThirdParty:
    NAME = "TTP"

    @staticmethod
    def generate_additive_triple(size0, size1, op, device=None, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        generator = TTPClient.get().get_generator(device=device)

        a = generate_random_ring_element(
            size0, generator=generator, device=device)
        b = generate_random_ring_element(
            size1, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            # Request c from TTP
            c = TTPClient.get().ttp_request(
                "additive", device, size0, size1, op, *args, **kwargs
            )
        else:
            # TODO: Compute size without executing computation
            c_size = getattr(torch, op)(a, b, *args, **kwargs).size()
            c = generate_random_ring_element(
                c_size, generator=generator, device=device)

        a = ArithmeticSharedTensor.from_shares(a, precision=0)
        b = ArithmeticSharedTensor.from_shares(b, precision=0)
        c = ArithmeticSharedTensor.from_shares(c, precision=0)

        return a, b, c

    @staticmethod
    def square(size, device=None):
        """Generate square double of given size"""
        generator = TTPClient.get().get_generator(device=device)

        r = generate_random_ring_element(
            size, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            # Request r2 from TTP
            r2 = TTPClient.get().ttp_request("square", device, size)
        else:
            r2 = generate_random_ring_element(
                size, generator=generator, device=device)

        r = ArithmeticSharedTensor.from_shares(r, precision=0)
        r2 = ArithmeticSharedTensor.from_shares(r2, precision=0)
        return r, r2

    @staticmethod
    def generate_binary_triple(size0, size1, device=None):
        """Generate binary triples of given size"""
        generator = TTPClient.get().get_generator(device=device)

        a = generate_kbit_random_tensor(
            size0, generator=generator, device=device)
        b = generate_kbit_random_tensor(
            size1, generator=generator, device=device)

        if comm.get().get_rank() == 0:
            # Request c from TTP
            c = TTPClient.get().ttp_request("binary", device, size0, size1)
        else:
            size2 = torch.broadcast_tensors(a, b)[0].size()
            c = generate_kbit_random_tensor(
                size2, generator=generator, device=device)

        # Stack to vectorize scatter function
        a = BinarySharedTensor.from_shares(a)
        b = BinarySharedTensor.from_shares(b)
        c = BinarySharedTensor.from_shares(c)
        return a, b, c

    @staticmethod
    def wrap_rng(size, device=None):
        """Generate random shared tensor of given size and sharing of its wraps"""
        generator = TTPClient.get().get_generator(device=device)

        r = generate_random_ring_element(
            size, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            # Request theta_r from TTP
            theta_r = TTPClient.get().ttp_request("wraps", device, size)
        else:
            theta_r = generate_random_ring_element(
                size, generator=generator, device=device
            )

        r = ArithmeticSharedTensor.from_shares(r, precision=0)
        theta_r = ArithmeticSharedTensor.from_shares(theta_r, precision=0)
        return r, theta_r

    @staticmethod
    def generate_fss_keys(rank, n_values, op):
        """
            Generate random bit tensor for fss, A pair of keys, one for rank 0, another for rank 1
            rank: rank of process
            n_values: number of key values
            op: eq or comp
        """
        size = (n_values, 621 if op == "eq" else 920)
        keys = torch.empty(size=size, dtype=torch.uint8)
        if rank == 0:

            TTPClient.get().fss_ttp_request(n_values, op)
            dist.recv(keys, src=2, group=TTPClient.get().ttp_group)
        else:
            dist.recv(keys, src=2, group=TTPClient.get().ttp_group)
        return keys

    # 使用可信第三方通信产生随机数
    @staticmethod
    def Get_Rand_Bit(size, device=None):
        """
            Generate random bit tensor and Arithmetic Tensor for securenn, one for rank 0, another for rank 1
        """
        rank = comm.get().get_rank()
        rand_val_temp = torch.empty(size=size, dtype=torch.int64)
        temp = (64,)
        bit_size = size + temp
        rand_val_bits_temp = torch.empty(size=bit_size, dtype=torch.int64)
        if rank == 0:
            TTPClient.get().Get_rand_bits_ttp_request("Get_Rand_Bit", device, size)
            dist.recv(rand_val_temp, src=comm.get().get_ttp_rank(),
                      group=TTPClient.get().ttp_group)
            dist.recv(rand_val_bits_temp, src=comm.get(
            ).get_ttp_rank(), group=TTPClient.get().ttp_group)
        else:
            dist.recv(rand_val_temp, src=comm.get().get_ttp_rank(),
                      group=TTPClient.get().ttp_group)
            dist.recv(rand_val_bits_temp, src=comm.get(
            ).get_ttp_rank(), group=TTPClient.get().ttp_group)
        # print(rand_val_bits_temp)
        rand_val = ArithmeticSharedTensor.from_shares(rand_val_temp)
        # rand_val = crypten.cryptensor(rand_val)
        rand_val_bits = ArithmeticSharedTensor(rand_val_bits_temp)
        # rand_val_bits = crypten.cryptensor(rand_val_bits)
        return rand_val, rand_val_bits

    @staticmethod
    def B2A_rng(size, device=None):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        generator = TTPClient.get().get_generator(device=device)

        # generate random bit
        rB = generate_kbit_random_tensor(
            size, bitlength=1, generator=generator, device=device
        )

        if comm.get().get_rank() == 0:
            # Request rA from TTP
            rA = TTPClient.get().ttp_request("B2A", device, size)
        else:
            rA = generate_random_ring_element(
                size, generator=generator, device=device)

        rA = ArithmeticSharedTensor.from_shares(rA, precision=0)
        rB = BinarySharedTensor.from_shares(rB)
        return rA, rB

    @staticmethod
    def rand(*sizes, encoder=None, device=None):
        """Generate random ArithmeticSharedTensor uniform on [0, 1]"""
        generator = TTPClient.get().get_generator(device=device)

        if isinstance(sizes, torch.Size):
            sizes = tuple(sizes)

        if isinstance(sizes[0], torch.Size):
            sizes = tuple(sizes[0])

        if comm.get().get_rank() == 0:
            # Request samples from TTP
            samples = TTPClient.get().ttp_request(
                "rand", device, *sizes, encoder=encoder
            )
        else:
            samples = generate_random_ring_element(
                sizes, generator=generator, device=device
            )
        return ArithmeticSharedTensor.from_shares(samples)

    @staticmethod
    def Securenn_PC_Reconst(d_enc, device=None):
        rank = comm.get().get_rank()
        # print(d_enc.get_plain_text())
        if rank == 0:
            size_one = TTPClient.get().PC_Reconst_ttp_request(
                "Securenn_PC_Reconst", device, d_enc.data)
            result = torch.empty(size=size_one, dtype=torch.int32)
            dist.recv(result, src=comm.get().get_ttp_rank(),
                      group=TTPClient.get().ttp_group)
            # dist.isend(tensor=d_enc.data, dst=2, group=TTPClient.get().ttp_group)
        else:
            size_two = TTPClient.get().PC_Reconst_ttp_request(
                "Securenn_PC_Reconst", device, d_enc.data)
            result = torch.empty(size=size_two, dtype=torch.int32)
            dist.recv(result, src=comm.get().get_ttp_rank(),
                      group=TTPClient.get().ttp_group)
            # dist.isend(tensor=d_enc.data, dst=2, group=TTPClient.get().ttp_group)
        # dist.recv(result, src=2, group=TTPClient.get().ttp_group)
        # print(result)
        ans = ArithmeticSharedTensor(result)
        # ans = crypten.cryptensor(ans)
        return ans

    @staticmethod
    # 通过可信第三方获取打乱的行的位置以及打乱的矩阵的各个元素的位置
    def Get_Shuffle_Matrix(n, device=None):
        rank = comm.get().get_rank()
        # 设置输出的数据的格式，这里是需要符合输出
        map_row_size = torch.Size([n, n])
        map_dict_size = torch.Size([n, n, n, n])
        map_row = torch.empty(size=map_row_size)
        map_dict = torch.empty(size=map_dict_size)
        # 不同的进程模拟不同的计算参与方，rank=0调用Get_Shuffle_Matrix_ttp_request请求接口，接口的参数（调用的函数名称，设备名称，函数参数）从而发起请求
        if rank == 0:
            TTPClient.get().Get_Shuffle_Matrix_ttp_request("Get_Shuffle_Matrix", device, n)
            # 计算参与方接受从可信地方返回的数据dist.recv（存放结果的容器（要提前设置好大小），源进程（就是接收从哪里发送过来的），group使用TTPClient.get().ttp_group）
            dist.recv(map_row, src=comm.get().get_ttp_rank(),
                      group=TTPClient.get().ttp_group)
            dist.recv(map_dict, src=comm.get().get_ttp_rank(),
                      group=TTPClient.get().ttp_group)
        else:
            dist.recv(map_row, src=comm.get().get_ttp_rank(),
                      group=TTPClient.get().ttp_group)
            dist.recv(map_dict, src=comm.get().get_ttp_rank(),
                      group=TTPClient.get().ttp_group)
        encoder = FixedPointEncoder(precision_bits=16)
        map_row = encoder.encode(map_row)
        map_row = ArithmeticSharedTensor.from_shares(map_row)
        map_dict = encoder.encode(map_dict)
        map_dict = ArithmeticSharedTensor.from_shares(map_dict)
        # print(map_row.get_plain_text())
        # print(map_dict.get_plain_text())
        return map_row, map_dict

    @staticmethod
    def Falcon_PC_Reconst(m, device=None):
        rank = comm.get().get_rank()
        # print(m.get_plain_text())
        if rank == 0:
            size_one = TTPClient.get().Falocn_PC_Reconst_ttp_request(
                "Falcon_PC_Reconst", device, m.data)
            result = torch.empty(size=size_one, dtype=torch.int32)
            dist.recv(result, src=comm.get().get_ttp_rank(),
                      group=TTPClient.get().ttp_group)
        else:
            size_two = TTPClient.get().Falocn_PC_Reconst_ttp_request(
                "Falcon_PC_Reconst", device, m.data)
            result = torch.empty(size=size_two, dtype=torch.int32)
            dist.recv(result, src=comm.get().get_ttp_rank(),
                      group=TTPClient.get().ttp_group)
        ans = ArithmeticSharedTensor(result)
        # print(ans.get_plain_text())
        return ans

    @staticmethod
    def Securenn_SC_Reconst(tilde_a, l):
        x = tilde_a.get_plain_text()
        delta = (x >= 2 ** l).int()
        x = x % (2 ** l)
        # print(x)
        # print(delta.type())
        enc_delta = ArithmeticSharedTensor(delta)
        x_bit = torch.empty((1), dtype=torch.int32)
        for i in range(l):
            if i == 0:
                x_bit = x % 2
                x //= 2
                x_bit = x_bit.unsqueeze(-1)
            else:
                mid = x % 2
                x //= 2
                mid = mid.unsqueeze(-1)
                x_bit = torch.cat((mid, x_bit), dim=-1)
        enc_x_bit = ArithmeticSharedTensor(x_bit)
        return enc_x_bit, enc_delta

    @staticmethod
    def Securenn_MSB_Generate(size, l):
        x_ = generate_random_positive_ring_element(
            size=size, ring_size=2 ** (l))
        x = ArithmeticSharedTensor.from_shares(x_)
        x_ = x.get_plain_text()
        sign = (x.get_plain_text() % 2).int()
        enc_x_sign = ArithmeticSharedTensor(sign)
        x_bit = torch.empty((1), dtype=torch.int32)
        for i in range(l):
            if i == 0:
                x_bit = x_ % 2
                x_ //= 2
                x_bit = x_bit.unsqueeze(-1)
            else:
                mid = x_ % 2
                x_ //= 2
                mid = mid.unsqueeze(-1)
                x_bit = torch.cat((mid, x_bit), dim=-1)
        enc_x_bit = ArithmeticSharedTensor(x_bit)
        # enc_x_sign=x[...,0].unsqueeze(-1)
        return x, enc_x_bit, enc_x_sign

    @staticmethod
    def Securenn_DMaxpool_Reconst(k, l, n):
        t = k.get_plain_text() % (2 ** l)
        k = t % n
        Ek = torch.empty((1), dtype=torch.int32)
        for i in range(n):
            if i == 0:
                Ek = (torch.randint(i, i + 1, k.size())
                      == k).int().unsqueeze(-1)
            else:
                mid = (torch.randint(i, i + 1, k.size())
                       == k).int().unsqueeze(-1)
                Ek = torch.cat((Ek, mid), dim=-1)
        Ek = Ek.int()
        Ek = ArithmeticSharedTensor(Ek)
        return Ek

    @staticmethod
    def testnn_Check(x: ArithmeticSharedTensor):
        zero = torch.zeros(x.share.shape)
        one = torch.ones(x.share.shape)
        x = x.get_plain_text()
        ans = torch.where(x > 0, one, zero)
        return ArithmeticSharedTensor(ans)

    @staticmethod
    def _init():
        TTPClient._init()

    @staticmethod
    def uninit():
        TTPClient.uninit()


class TTPClient:
    __instance = None

    class __TTPClient:
        """Singleton class"""

        def __init__(self):
            # Initialize connection
            self.ttp_group = comm.get().ttp_group
            self.comm_group = comm.get().ttp_comm_group
            self._setup_generators()
            logging.info(f"TTPClient {comm.get().get_rank()} initialized")

        def _setup_generators(self):
            """Setup RNG generator shared between each party (client) and the TTPServer"""
            seed = torch.empty(size=(), dtype=torch.long)
            dist.irecv(
                tensor=seed, src=comm.get().get_ttp_rank(), group=self.ttp_group
            ).wait()
            dist.barrier(group=self.ttp_group)

            self.generator = torch.Generator(device="cpu")
            self.generator.manual_seed(seed.item())

            if torch.cuda.is_available():
                self.generator_cuda = torch.Generator(device="cuda")
                self.generator_cuda.manual_seed(seed.item())
            else:
                self.generator_cuda = None

        def get_generator(self, device=None):
            if device is None:
                device = "cpu"
            device = torch.device(device)
            if device.type == "cuda":
                return self.generator_cuda
            else:
                return self.generator

        # 实际上是一个client的请求接口，设置请求的函数名，参数等
        def ttp_request(self, func_name, device, *args, **kwargs):
            assert (
                comm.get().get_rank() == 0
            ), "Only party 0 communicates with the TTPServer"
            if device is not None:
                device = str(device)
            message = {
                "function": func_name,
                "device": device,
                "args": args,
                "kwargs": kwargs,
            }
            ttp_rank = comm.get().get_ttp_rank()

            comm.get().send_obj(message, ttp_rank, self.ttp_group)

            size = comm.get().recv_obj(ttp_rank, self.ttp_group)
            result = torch.empty(size, dtype=torch.long, device=device)
            comm.get().broadcast(result, ttp_rank, self.comm_group)
            return result

        def Get_rand_bits_ttp_request(self, func_name, device, *args, **kwargs):
            assert (
                comm.get().get_rank() == 0
            ), "Only party 0 communicates with the TTPServer"
            if device is not None:
                device = str(device)
            message = {
                "function": func_name,
                "device": device,
                "args": args,
                "kwargs": kwargs,
            }
            ttp_rank = comm.get().get_ttp_rank()

            comm.get().send_obj(message, ttp_rank, self.ttp_group)

        def PC_Reconst_ttp_request(self, func_name, device, *args, **kwargs):
            if device is not None:
                device = str(device)
            message = {
                "function": func_name,
                "device": device,
                "args": args,
                "kwargs": kwargs,
            }
            ttp_rank = comm.get().get_ttp_rank()

            comm.get().send_obj(message, ttp_rank, self.ttp_group)
            size = comm.get().recv_obj(ttp_rank, self.ttp_group)
            return size

        def Falocn_PC_Reconst_ttp_request(self, func_name, device, *args, **kwargs):
            if device is not None:
                device = str(device)
            message = {
                "function": func_name,
                "device": device,
                "args": args,
                "kwargs": kwargs,
            }
            ttp_rank = comm.get().get_ttp_rank()

            comm.get().send_obj(message, ttp_rank, self.ttp_group)
            size = comm.get().recv_obj(ttp_rank, self.ttp_group)
            return size

        # 这就是对应的请求接口，这里写成了只接受rank0的请求，也可以去掉。
        def Get_Shuffle_Matrix_ttp_request(self, func_name, device, *args, **kwargs):
            assert (
                comm.get().get_rank() == 0
            ), "Only party 0 communicates with the TTPServer"
            if device is not None:
                device = str(device)
            message = {
                "function": func_name,
                "device": device,
                "args": args,
                "kwargs": kwargs,
            }
            # 获取第三方所在的进程，并将信息发送过去，服务器端是一直运行的，等待客户端的请求，当请求发送后服务端就会处理
            ttp_rank = comm.get().get_ttp_rank()
            comm.get().send_obj(message, ttp_rank, self.ttp_group)

        def fss_ttp_request(self, *args, **kwargs):
            assert (
                comm.get().get_rank() == 0
            ), "Only party 0 communicates with the TTPServer"

            message = {
                "function": "fss",
                "device": "",
                "args": args,
                "kwargs": kwargs,
            }
            ttp_rank = comm.get().get_ttp_rank()

            comm.get().send_obj(message, ttp_rank, self.ttp_group)

    @staticmethod
    def _init():
        """Initializes a Trusted Third Party client that sends requests"""
        if TTPClient.__instance is None:
            TTPClient.__instance = TTPClient.__TTPClient()

    @staticmethod
    def uninit():
        """Uninitializes a Trusted Third Party client"""
        del TTPClient.__instance
        TTPClient.__instance = None

    @staticmethod
    def get():
        """Returns the instance of the TTPClient"""
        if TTPClient.__instance is None:
            raise RuntimeError("TTPClient is not initialized")

        return TTPClient.__instance


N_CORES = multiprocessing.cpu_count()
dpf = sycret.EqFactory(n_threads=N_CORES)
dif = sycret.LeFactory(n_threads=N_CORES)


class TTPServer:
    TERMINATE = -1

    def __init__(self):
        """Initializes a Trusted Third Party server that receives requests"""
        # Initialize connection
        crypten.init()
        self.ttp_group = comm.get().ttp_group
        self.comm_group = comm.get().ttp_comm_group
        self.device = "cpu"
        self._setup_generators()
        ttp_rank = comm.get().get_ttp_rank()

        logging.info("TTPServer Initialized")
        try:
            while True:
                # Wait for next request from client
                message = comm.get().recv_obj(0, self.ttp_group)
                logging.info("Message received: %s" % message)

                if message == "terminate":
                    logging.info("TTPServer shutting down.")
                    return

                function = message["function"]
                device = message["device"]
                args = message["args"]
                kwargs = message["kwargs"]

                # fss密钥产生不能使用crypten自带的，所以通信也有差异，所以重写添加如下内容
                if function == "fss":
                    keys = self.fss(*args, **kwargs)
                    reqs = [
                        dist.isend(tensor=keys[i], dst=i, group=self.ttp_group) for i in range(2)
                    ]
                    for req in reqs:
                        req.wait()
                    continue

                self.device = device

                if function == "Get_Rand_Bit":
                    r_one, r_two, bit = getattr(
                        self, function)(*args, **kwargs)
                    r = [r_one, r_two]
                    b = bit
                    reqs = [
                        dist.isend(tensor=r[i], dst=i, group=self.ttp_group) for i in range(2)
                    ]
                    for req in reqs:
                        req.wait()
                    reqss = [
                        dist.isend(tensor=b, dst=i, group=self.ttp_group) for i in range(2)
                    ]
                    for req in reqss:
                        req.wait()
                    continue
                # 服务端接收到客户端的请求，从请求执行的函数名分成不同的情况执行
                if function == "Get_Shuffle_Matrix":
                    # print(comm.get().get_rank())
                    # 执行函数获取结果
                    map_row_one, map_row_two, map_dict_one, map_dict_two = getattr(
                        self, function)(*args, **kwargs)
                    map_row = [map_row_one, map_row_two]
                    map_dict = [map_dict_one, map_dict_two]
                    # print(map_row[0],map_dict[0])
                    # 发送结果
                    reqs = [
                        dist.isend(tensor=map_row[i], dst=i, group=self.ttp_group) for i in range(2)
                    ]
                    for req in reqs:
                        req.wait()
                    reqss = [
                        dist.isend(tensor=map_dict[i], dst=i, group=self.ttp_group) for i in range(2)
                    ]
                    for req in reqss:
                        req.wait()
                    continue

                if function == "Securenn_PC_Reconst":
                    message_two = comm.get().recv_obj(1, self.ttp_group)
                    d_enc_one = args[0]
                    d_enc_two = message_two["args"][0]
                    encoder = FixedPointEncoder(precision_bits=16)
                    d_enc = encoder.decode(d_enc_one + d_enc_two)
                    # print(d_enc)
                    result = getattr(self, function)(d_enc)
                    size = result.size()
                    comm.get().send_obj(size, 0, self.ttp_group)
                    comm.get().send_obj(size, 1, self.ttp_group)
                    reqs = [
                        dist.isend(tensor=result, dst=i, group=self.ttp_group) for i in range(2)
                    ]
                    for req in reqs:
                        req.wait()
                    continue

                if function == "Falcon_PC_Reconst":
                    message_two = comm.get().recv_obj(1, self.ttp_group)
                    m_one = args[0]
                    m_two = message_two["args"][0]
                    encoder = FixedPointEncoder(precision_bits=16)
                    m = encoder.decode(m_one + m_two)
                    result = getattr(self, function)(m)
                    size = result.size()
                    comm.get().send_obj(size, 0, self.ttp_group)
                    comm.get().send_obj(size, 1, self.ttp_group)
                    reqs = [
                        dist.isend(tensor=result, dst=i, group=self.ttp_group) for i in range(2)
                    ]
                    for req in reqs:
                        req.wait()
                    continue

                result = getattr(self, function)(*args, **kwargs)
                size = result.size()
                comm.get().send_obj(size, 0, self.ttp_group)
                comm.get().broadcast(result, ttp_rank, self.comm_group)
        except RuntimeError as err:
            logging.info("Encountered Runtime error. TTPServer shutting down:")
            logging.info(f"{err}")

    def _setup_generators(self):
        """Create random generator to send to a party"""
        ws = comm.get().get_world_size()

        seeds = [torch.randint(-(2 ** 63), 2 ** 63 - 1, size=())
                 for _ in range(ws)]
        reqs = [
            dist.isend(tensor=seeds[i], dst=i, group=self.ttp_group) for i in range(ws)
        ]
        self.generators = [torch.Generator(device="cpu") for _ in range(ws)]
        self.generators_cuda = [
            (torch.Generator(device="cuda") if torch.cuda.is_available() else None)
            for _ in range(ws)
        ]

        for i in range(ws):
            self.generators[i].manual_seed(seeds[i].item())
            if torch.cuda.is_available():
                self.generators_cuda[i].manual_seed(seeds[i].item())
            reqs[i].wait()

        dist.barrier(group=self.ttp_group)

    def _get_generators(self, device=None):
        if device is None:
            device = "cpu"
        device = torch.device(device)
        if device.type == "cuda":
            return self.generators_cuda
        else:
            return self.generators

    def _get_additive_PRSS(self, size, remove_rank=False):
        """
        Generates a plaintext value from a set of random additive secret shares
        generated by each party
        """
        gens = self._get_generators(device=self.device)
        if remove_rank:
            gens = gens[1:]
        result = None
        for idx, g in enumerate(gens):
            elem = generate_random_ring_element(
                size, generator=g, device=g.device)
            result = elem if idx == 0 else result + elem
        return result

    def _get_binary_PRSS(self, size, bitlength=None, remove_rank=None):
        """
        Generates a plaintext value from a set of random binary secret shares
        generated by each party
        """
        gens = self._get_generators(device=self.device)
        if remove_rank:
            gens = gens[1:]
        result = None
        for idx, g in enumerate(gens):
            elem = generate_kbit_random_tensor(
                size, bitlength=bitlength, generator=g, device=g.device
            )
            result = elem if idx == 0 else result ^ elem
        return result

    def fss(self, n_values, op="eq"):
        """
        Generates keys for fss
        """
        if op == "eq":
            primitives = dpf.keygen(n_values=n_values)
        elif op == "comp":
            primitives = dif.keygen(n_values=n_values)
        else:
            raise ValueError(f"{op} is an FSS unsupported operation.")

        keys = [torch.tensor(p) for p in primitives]
        return keys

    def A2B(self, x_):
        l = 64
        for i in range(l):
            if i == 0:
                x_bit = x_ % 2
                x_ //= 2
                x_bit = x_bit.unsqueeze(-1)
            else:
                mid = x_ % 2
                x_ //= 2
                mid = mid.unsqueeze(-1)
                x_bit = torch.cat((mid, x_bit), dim=-1)
        return x_bit

    def Get_Rand_Bit(self, size):
        l = 30
        # r = generate_random_positive_ring_element(size,generator=self.generators)
        # generator = torch.Generator("cpu")
        encoder = FixedPointEncoder(precision_bits=16)
        r_one = torch.randint(1, 2 ** (l - 1), size=size, dtype=torch.int64)
        r_two = torch.randint(1, 2 ** (l - 7), size=size, dtype=torch.int64)
        r_one = encoder.encode(r_one)
        r_two = encoder.encode(r_two)
        # rr = ArithmeticSharedTensor.from_shares(r)
        x = r_one + r_two
        bit = self.A2B(x)
        r_now_one = r_one.clone()
        r_now_two = r_two.clone()
        # bit_one = self.A2B(r_one)
        # bit_two = self.A2B(r_two)
        return r_now_one, r_now_two, bit

    def Securenn_PC_Reconst(self, d_enc):
        # 揭露d
        # print("I'm here")
        device = None
        l = 64
        list_ = d_enc.int()
        list_ = (list_ != 0).int()
        # 用连乘方法验证0的存在
        ans = torch.ones((d_enc[..., 0].size()))
        for i in range(l):
            ans = ans * list_[..., i]
        ans = (ans == 0).int()
        return ans

    def Falcon_PC_Reconst(self, m):
        # print(m)
        ans = (m != 0).int()
        # print(ans)
        return ans

    def Get_Shuffle_Matrix(self, n):
        # TTP 执行
        pai_i = torch.from_numpy(np.random.permutation(n))
        pai_j = torch.from_numpy(np.random.permutation(n))
        map_row_size = torch.Size([n, n])
        map_dict_size = torch.Size([n, n, n, n])
        map_row = torch.zeros(size=map_row_size, dtype=torch.int64)
        map_dict = torch.zeros(size=map_dict_size, dtype=torch.int64)
        for i in range(n):
            map_row[i][pai_i[i]] = 1
            for j in range(n):
                temp = torch.zeros(size=map_row_size)
                temp[pai_i[i]][pai_j[j]] = 1
                map_dict[i][j] = temp
        map_row_one = torch.ceil(torch.rand(size=map_row_size)*(2**10))
        map_dict_one = torch.ceil(torch.rand(size=map_dict_size)*(2**10))
        map_row_two = map_row - map_row_one
        map_dict_two = map_dict - map_dict_one
        return map_row_one, map_row_two, map_dict_one, map_dict_two

    def additive(self, size0, size1, op, *args, **kwargs):

        # Add all shares of `a` and `b` to get plaintext `a` and `b`
        a = self._get_additive_PRSS(size0)
        b = self._get_additive_PRSS(size1)

        c = getattr(torch, op)(a, b, *args, **kwargs)

        # Subtract all other shares of `c` from plaintext value of `c` to get `c0`
        c0 = c - self._get_additive_PRSS(c.size(), remove_rank=True)
        return c0

    def square(self, size):
        # Add all shares of `r` to get plaintext `r`
        r = self._get_additive_PRSS(size)
        r2 = r.mul(r)
        return r2 - self._get_additive_PRSS(size, remove_rank=True)

    def binary(self, size0, size1):
        # xor all shares of `a` and `b` to get plaintext `a` and `b`
        a = self._get_binary_PRSS(size0)
        b = self._get_binary_PRSS(size1)

        c = a & b

        # xor all other shares of `c` from plaintext value of `c` to get `c0`
        c0 = c ^ self._get_binary_PRSS(c.size(), remove_rank=True)
        return c0

    def wraps(self, size):
        r = [generate_random_ring_element(
            size, generator=g) for g in self.generators]
        theta_r = count_wraps(r)

        return theta_r - self._get_additive_PRSS(size, remove_rank=True)

    def B2A(self, size):
        rB = self._get_binary_PRSS(size, bitlength=1)

        # Subtract all other shares of `rA` from plaintext value of `rA`
        rA = rB - self._get_additive_PRSS(size, remove_rank=True)

        return rA
