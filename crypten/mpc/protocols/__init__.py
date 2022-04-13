from .crypten import CrypTenProtocol
from .malicious import MaliciousProtocol
from .test import TestProtocol
from .fss import FSSProtocol
from .sercureNN import SecureNNProtocol
from .spdz import SPDZProtocol
from .falcon import FalconProtocol
__all__ = [
    "CrypTenProtocol",
    "ActivateProtocol"
]

__SUPPORTED_PROTOCOLS = {
    "CRYPTEN": CrypTenProtocol,
    "SECURENN": SecureNNProtocol,
    "FSS": FSSProtocol,
    "TEST": TestProtocol,
    "MALICIOUS":MaliciousProtocol,
    "SPDZ": SPDZProtocol,
    "FALCON": FalconProtocol,
}
# Set default protocol
ActivateProtocol = __SUPPORTED_PROTOCOLS["CRYPTEN"]

def set_activate_protocol(protocol):
    """ 用于设置默认协议类型 """
    global ActivateProtocol
    assert_msg = "Protocol %s is not supported" % protocol
    if isinstance(protocol, str):
        protocol = protocol.upper()
        assert protocol in __SUPPORTED_PROTOCOLS.keys(), assert_msg
        protocol = __SUPPORTED_PROTOCOLS[protocol]
    else:
        assert protocol in __SUPPORTED_PROTOCOLS.values(), assert_msg
    ActivateProtocol = protocol


def get_activate_protocol():
    return ActivateProtocol
