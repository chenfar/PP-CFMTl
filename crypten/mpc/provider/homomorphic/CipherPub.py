class CipherPub(object):
    def __init__(self, T1=None, T2=None, PUB=None, exponent=0):
        self.T1 = T1
        self.T2 = T2
        self.PUB = PUB

    def __add__(self, other):
        if isinstance(other, CipherPub):
            return self._add_encrypted(other)

    def __radd__(self, other):
        return self.__add__(other)

    def _add_encrypted(self, other):
        if self.PUB != other.PUB:
            raise ValueError("Attempted to add numbers encrypted against "
                             "different public keys!")
        a, b = self, other
        T1 = a.T1 * b.T1
        T2 = a.T2 * b.T2
        pub = a.PUB
        return CipherPub(T1, T2, pub)
