from crypten.mpc.provider.homomorphic.hm_util import invert, powmod, getprimeover
from crypten.mpc.provider.homomorphic.CipherPub import CipherPub
from crypten.mpc.provider.homomorphic.Ciphertext1 import Ciphertext1

import torch
import random
import gmpy2 as _g

try:
    from collections.abc import Mapping
except ImportError:
    Mapping = dict

DEFAULT_KEYSIZE = 32


def divv(pair):
    a, b = pair
    return a / b


def randomPrime(n_length):
    p = q = n = None
    n_len = 0
    while n_len != n_length:
        p = getprimeover(n_length // 2)
        q = p
        while q == p:
            q = getprimeover(n_length // 2)
        n = p * q  # 求公钥n
        n_len = n.bit_length()
        # .bit_length()  # 计算n的二进制长度
    if (p == q):
        raise ValueError('p and q have to be different')
    return p, q


def lcm(p_, q_):
    return (p_ * q_) // _g.gcd(p_, q_)


def generate_paillier_keypair(n_length=DEFAULT_KEYSIZE):
    p, q = randomPrime(n_length)
    a = getprimeover(n_length // 2)
    x = getprimeover(n_length // 2)
    public_key = PaillierPublicKey(n_length, a, x, p, q)
    private_key = PaillierPrivateKey(public_key, n_length, a, x, p, q)
    return public_key, private_key


class PaillierPublicKey():
    def __init__(self, bitLengthVal=1024, a=None, x=None, p=None, q=None):
        self.alpha = 2
        self.beta = 3
        self.bitLengthVal = bitLengthVal
        self.a = a
        self.N = p * q
        self.N2 = self.N ** 2
        self.g1 = 2

        self.g = (-powmod(self.a, self.g1 * self.N, self.N2)) % self.N2
        self.x = x
        self.h = powmod(self.g, self.x, self.N2)
        self.X = [None] * self.beta
        self.H = [None] * self.beta
        self.lambda1 = [None] * self.alpha
        self.Xsigma = 0
        self.Hsigma = powmod(self.g, self.Xsigma, self.N2)
        for ii in range(0, self.beta):
            self.X[ii] = getprimeover(self.bitLengthVal - 12)
            self.H[ii] = powmod(self.g, self.X[ii], self.N2)
            self.Xsigma += self.X[ii]

        self.lamda = (p - 1) * (q - 1) // _g.gcd(p - 1, q - 1)
        self.KK1 = self.lamda * self.N2

        self.KKK = invert(_g.mpz(self.lamda), _g.mpz(self.N2))
        self.S = (self.lamda * self.KKK) % self.KK1
        self.lambda1[self.alpha - 1] = self.S

        for ii in range(0, self.alpha - 1):
            self.lambda1[ii] = getprimeover(self.bitLengthVal)
            self.lambda1[self.alpha - 1] = self.lambda1[self.alpha - 1] - self.lambda1[ii]

    def Encrypt(self, plaintext, *args):
        r = random.SystemRandom().randrange(1, 2 ** self.bitLengthVal)
        # if isinstance(plaintext, torch.Tensor):
        if args and len(args) > 0:
            cc = CipherPub()
            h = args[0]
            cc.PUB = h
        else:
            cc = CipherPub()
            cc.PUB = self.h
            h = cc.PUB

        cc.T1 = (((1 + plaintext * self.N) % self.N2) * powmod(h, r, self.N2)) % self.N2
        cc.T2 = powmod(self.g, r, self.N2)
        return cc

    def Refreash(self, c, *args):  # c Ciphertext hp integer
        cc = CipherPub()
        r = random.SystemRandom().randrange(1, 2 ** self.bitLengthVal)
        cc.T1 = (c.T1 * (powmod(c.PUB, r, self.N2))) % self.N2
        cc.T2 = (c.T2 * (powmod(self.g, r, self.N2))) % self.N2
        cc.PUB = c.PUB
        return cc

    def Encrypt_tourch(self, tensor, *args):
        size = tensor.size()
        list = tensor.numpy().tolist()

        return 0

class PaillierPrivateKey():

    def __init__(self, public_key, bitLengthVal=1024, a=None, x=None, p=None, q=None):
        self.public_key = public_key
        self.alpha = 2
        self.beta = 3
        self.bitLengthVal = bitLengthVal

        self.lambda1 = self.public_key.lambda1
        self.N = p * q
        self.g1 = 2
        self.N2 = self.N ** 2
        self.a = a
        self.g = (-powmod(self.a, self.g1 * self.N, self.N2)) % self.N2
        self.g1 = 2
        self.x = x

        self.x1 = getprimeover(self.bitLengthVal // 4)
        self.lamda = (p - 1) * (q - 1) // _g.gcd(p - 1, q - 1)

        self.Hsigma = powmod(self.g, self.public_key.Xsigma, self.N2)
        self.h = powmod(self.g, self.x, self.N2)
        self.x2 = self.x - self.x1

    def SDecryption(self, c):  # c ciphertext
        if isinstance(c, CipherPub):
            u1 = invert(_g.mpz(self.lamda), _g.mpz(self.N))
            return (((powmod(int(c.T1), int(self.lamda), self.N2) - 1) // self.N) * u1) % self.N

    def AddPDec1(self, c, *args):  # c是 ciphertext hp integer
        if isinstance(c, CipherPub):
            cc = Ciphertext1()
            r = random.SystemRandom().randrange(1, 2 ** self.bitLengthVal)
            cc.T1 = (c.T1 * powmod(c.PUB, r, self.N2)) % self.N2
            cc.T2 = (c.T2 * powmod(self.g, r, self.N2)) % self.N2
            cc.T3 = powmod(cc.T1, self.lambda1[0], self.N2)
            return cc
        else:
            cc = Ciphertext1()
            r = random.SystemRandom().randrange(1, 2 ** self.bitLengthVal)
            cc.T1 = (c.T1 * powmod(self.Hsigma, r, self.N2)) % self.N2
            cc.T2 = (c.T2 * powmod(self.g, r, self.N2)) % self.N2
            cc.T3 = powmod(cc.T1, self.lambda1[0], self.N2)
            return cc

    def AddPDec2(self, c):  # c Ciphertext1
        cc = ((powmod(int(c.T1), int(self.lambda1[1]), self.N2) * c.T3)) % self.N2
        return ((cc - 1) // self.N) % self.N

    def Decrypt(self, ciphertext):
        ciphertext1 = self.AddPDec1(ciphertext)
        result = self.AddPDec2(ciphertext1)
        return result


def getnewList(newlist):
    d = []
    for element in newlist:
        if not isinstance(element, list):
            d.append(element)
        else:
            d.extend(getnewList(element))
    return d


if __name__ == "__main__":
    public_key, private_key = generate_paillier_keypair(1024)
    plaintext = 4
    encrypt_num = public_key.Encrypt(plaintext)

    # print(encrypt_num.T1)
    # print(encrypt_num.T2)
    # print(encrypt_num.PUB)
    # encrypt_tensor = public_key.Encrypt_tourch(tensor)
    tensor = torch.Tensor(
        [[[1, 2, 3], [1, 2, 3]], [[1, 2, 5], [1, 2, 5]]]
    )
    print(tensor.size())
    list1 = tensor.numpy().tolist()
    print(list1)
    list1 = getnewList(list1)
    print(list1)
    # encrypt_num1 = public_key.Encrypt(234)
    # encrypt_num = encrypt_num + encrypt_num1
    # print(private_key.SDecryption(encrypt_num))
    # print(private_key.AddPDec2(private_key.AddPDec1(encrypt_num)))
