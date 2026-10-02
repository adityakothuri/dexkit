"""Byte-for-byte parity between our packet encoder and Feetech's scservo_sdk (hls class).

The mocks share our assumptions; this checks us against the vendor implementation.
"""

import pytest

scs = pytest.importorskip("scservo_sdk")

from dexkit.hw import feetech_protocol as fp  # noqa: E402


class CapturePort:
    """Minimal PortHandler stand-in: records TX, never answers (SDK calls time out harmlessly)."""

    is_using = False

    def __init__(self):
        self.tx = []
        self.baudrate = 1_000_000
        self.ser = None

    def clearPort(self):
        pass

    def writePort(self, packet):
        self.tx.append(bytes(packet))
        return len(packet)

    def readPort(self, length):
        return []

    def setPacketTimeout(self, *_):
        pass

    def setPacketTimeoutMillis(self, *_):
        pass

    def isPacketTimeout(self):
        return True

    def getBaudRate(self):
        return self.baudrate


@pytest.fixture
def sdk():
    port = CapturePort()
    return port, scs.hls(port)


def test_ping_parity(sdk):
    port, ph = sdk
    ph.ping(7)
    assert port.tx[0] == fp.ping_packet(7)


def test_read_parity(sdk):
    port, ph = sdk
    ph.read2ByteTxRx(3, 56)
    assert port.tx[0] == fp.read_packet(3, 56, 2)


def test_write_torque_and_goal_torque_parity(sdk):
    port, ph = sdk
    ph.write1ByteTxRx(5, 40, 1)
    ph.write2ByteTxRx(5, 44, 600)
    assert port.tx[0] == fp.write_packet(5, 40, [1])
    assert port.tx[1] == fp.write_packet(5, 44, fp.le16(600))


def test_sync_write_goal_position_parity(sdk):
    port, ph = sdk
    gsw = scs.GroupSyncWrite(ph, 42, 2)
    targets = {1: 2048, 2: 1900, 13: 3000}
    for sid, t in targets.items():
        gsw.addParam(sid, [ph.scs_lobyte(t), ph.scs_hibyte(t)])
    gsw.txPacket()
    ours = fp.sync_write_packet(42, 2, {sid: fp.encode_position(t) for sid, t in targets.items()})
    assert port.tx[0] == ours


def test_sync_read_parity(sdk):
    port, ph = sdk
    gsr = scs.GroupSyncRead(ph, 56, 8)
    for sid in (1, 2, 3, 13):
        gsr.addParam(sid)
    gsr.txPacket()
    assert port.tx[0] == fp.sync_read_packet(56, 8, [1, 2, 3, 13])


def test_position_sign_convention_matches_sdk(sdk):
    _, ph = sdk
    for v in (0, 2048, 4095):
        raw = fp.from_le16(*fp.encode_position(v))
        assert ph.scs_tohost(raw, 15) == v
    raw = fp.from_le16(*fp.encode_position(-100))
    assert ph.scs_tohost(raw, 15) == -100
