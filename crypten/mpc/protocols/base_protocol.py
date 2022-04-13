"""
协议基类，包括乘法类操作：mul，matmul，conv1d，conv2d，conv_transpose1d，conv_transpose2d
            比较类操作：ltz，eq，ne

问题：有的协议做比较是比较 x 和 y，但是crypten比较接口全部依赖 x 和 0 比较。
方案：要么修改协议，必须实现x和0比较，要不修改crypten，但是要改的地方太多。
"""


class BaseProtocol(object):
    @staticmethod
    def mul(x, y):
        raise NotImplementedError("mul is not implemented")

    @staticmethod
    def matmul(x, y):
        raise NotImplementedError("matmul is not implemented")

    @staticmethod
    def conv1d(x, y, **kwargs):
        raise NotImplementedError("conv1d is not implemented")

    @staticmethod
    def conv2d(x, y, **kwargs):
        raise NotImplementedError("conv2d is not implemented")

    @staticmethod
    def conv_transpose1d(x, y, **kwargs):
        raise NotImplementedError("conv_transpose1d is not implemented")

    @staticmethod
    def conv_transpose2d(x, y, **kwargs):
        raise NotImplementedError("conv_transpose2d is not implemented")

    # Comparators
    @staticmethod
    def ltz(x):
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        raise NotImplementedError("ltz is not implemented")

    @staticmethod
    def lt(x, y):
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        raise NotImplementedError("ltz is not implemented")

    @staticmethod
    def le(x, y):
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        raise NotImplementedError("ltz is not implemented")

    @staticmethod
    def gt(x, y):
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        raise NotImplementedError("ltz is not implemented")

    @staticmethod
    def ge(x, y):
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        raise NotImplementedError("ltz is not implemented")

    @staticmethod
    def eq(x, y):
        """Returns x == y"""
        raise NotImplementedError("eq is not implemented")

    @staticmethod
    def ne(x, y):
        """Returns x != y"""
        raise NotImplementedError("ne is not implemented")

    @staticmethod
    def select_share(x, y, a):
        """Returns (1-a)x+ay"""
        raise NotImplementedError("ne is not implemented")
