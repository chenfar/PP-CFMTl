#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import multiprocessing

import sycret

import crypten.communicator as comm
import torch
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.util import count_wraps, torch_stack
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor

N_CORES = multiprocessing.cpu_count()
dpf = sycret.EqFactory(n_threads=N_CORES)
dif = sycret.LeFactory(n_threads=N_CORES)

class HomomorphicProvider:
    NAME = "HE"

    @staticmethod
    def generate_additive_triple(size0, size1, op, device=None, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""

        a = generate_random_ring_element(size0, device=device)
        b = generate_random_ring_element(size1, device=device)

        c = getattr(torch, op)(a, b, *args, **kwargs)

        a = ArithmeticSharedTensor(a, precision=0, src=0)
        b = ArithmeticSharedTensor(b, precision=0, src=0)
        c = ArithmeticSharedTensor(c, precision=0, src=0)

        return a, b, c



    @staticmethod
    def square(size):
        """Generate square double of given size"""
        raise NotImplementedError("HomomorphicProvider not implemented")

    @staticmethod
    def generate_xor_triple(size0, size1):
        """Generate xor triples of given size"""
        raise NotImplementedError("HomomorphicProvider not implemented")

    @staticmethod
    def wrap_rng(size, num_parties):
        """Generate random shared tensor of given size and sharing of its wraps"""
        raise NotImplementedError("HomomorphicProvider not implemented")

    @staticmethod
    def B2A_rng(size):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        raise NotImplementedError("HomomorphicProvider not implemented")
