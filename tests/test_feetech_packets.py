import pytest

from dexkit.hw.feetech_protocol import (
    INST_SYNC_WRITE,
    checksum,
    decode_load,
    decode_position,
    decode_status,
    encode_position,
    from_le16,
    le16,
    ping_packet,
    read_packet,
    sync_read_packet,
    sync_write_packet,
    write_packet,
)


def test_ping_matches_feetech_documented_example():
    # Feetech protocol manual: PING ID 1 -> FF FF 01 02 01 FB
    assert ping_packet(1) == bytes.fromhex("FF FF 01 02 01 FB")


def test_checksum_is_inverted_low_byte_of_sum():
    assert checksum([0x01, 0x02, 0x01]) == 0xFB
    assert checksum([0xFE, 0xFF, 0xFF]) == (~(0xFE + 0xFF + 0xFF)) & 0xFF


def test_read_present_position_packet():
    # READ id 1, address 56, 2 bytes: sum = 1+4+2+56+2 = 65 -> ~65 & 0xFF = 0xBE
    assert read_packet(1, 56, 2) == bytes.fromhex("FF FF 01 04 02 38 02 BE")


def test_write_torque_enable_packet():
    # WRITE id 1, addr 40, value 1: sum = 1+4+3+40+1 = 49 -> 0xCE
    assert write_packet(1, 40, [1]) == bytes.fromhex("FF FF 01 04 03 28 01 CE")


def test_sync_write_two_servos_hand_computed():
    # Goal position (addr 42, 2 bytes): id1 -> 2048 (00 08), id2 -> 1024 (00 04)
    pkt = sync_write_packet(42, 2, {1: encode_position(2048), 2: encode_position(1024)})
    body = [0xFE, 0x0A, INST_SYNC_WRITE, 42, 2, 1, 0x00, 0x08, 2, 0x00, 0x04]
    assert len(body) - 3 + 2 == 0x0A  # LEN = (L+1)*N + 4 = 3*2 + 4
    expected = bytes([0xFF, 0xFF] + body + [(~sum(body)) & 0xFF])
    assert pkt == expected
    assert pkt.hex(" ").upper() == "FF FF FE 0A 83 2A 02 01 00 08 02 00 04 " + f"{expected[-1]:02X}"


def test_sync_read_packet_layout():
    pkt = sync_read_packet(56, 8, [1, 2, 3])
    assert pkt[:5] == bytes([0xFF, 0xFF, 0xFE, 0x07, 0x82])
    assert list(pkt[5:10]) == [56, 8, 1, 2, 3]


def test_little_endian_round_trip():
    for v in (0, 1, 255, 256, 2048, 4095, 0x7FFF):
        lo, hi = le16(v)
        assert from_le16(lo, hi) == v
        assert decode_position(*encode_position(v)) == v
    assert le16(0x1234) == [0x34, 0x12]  # low byte first (STS/HLS), not SCS big-endian


def test_negative_position_uses_sign_bit():
    assert encode_position(-100) == le16(100 | 0x8000)
    assert decode_position(*encode_position(-100)) == -100


def test_load_magnitude_ignores_direction_bit():
    assert decode_load(*le16(500)) == 500
    assert decode_load(*le16(500 | 0x400)) == 500


def test_decode_status_validates_checksum():
    body = [1, 4, 0, 0x00, 0x08]
    pkt = bytes([0xFF, 0xFF] + body + [checksum(body)])
    sid, err, params = decode_status(pkt)
    assert (sid, err, params) == (1, 0, b"\x00\x08")
    with pytest.raises(IOError):
        decode_status(pkt[:-1] + bytes([pkt[-1] ^ 0xFF]))


def test_out_of_range_inputs_rejected():
    with pytest.raises(ValueError):
        le16(70000)
    with pytest.raises(ValueError):
        write_packet(1, 300, [0])
    with pytest.raises(ValueError):
        sync_write_packet(42, 2, {1: [1, 2, 3]})


def test_driver_reads_and_writes_against_mock_bus(hand_cfg):
    from dexkit.hw.feetech_hand import FeetechDriver
    from dexkit.hw.feetech_protocol import FeetechBus
    from dexkit.hw.mock import MOCK_MODEL_ROLL, MockFeetechSerial, mock_servos_for

    bus = MockFeetechSerial(mock_servos_for(hand_cfg))
    d = FeetechDriver(FeetechBus(bus), hand_cfg.register_map())
    assert d.ping(13) == MOCK_MODEL_ROLL
    assert d.ping(99) is None
    assert d.read_voltage(1) == pytest.approx(7.4)
    assert d.read_positions([1, 2, 3]) == {1: 2048, 2: 2048, 3: 2048}
    assert bus.packets(0x82), "sync read should be used for read_positions"
    d.sync_write_positions({1: 2100, 2: 2000})
    assert bus.servos[1].goal == 2100 and bus.servos[2].goal == 2000
    block = d.read_state_block([1, 2])
    assert set(block) == {1, 2} and block[1]["voltage"] == pytest.approx(7.4)
