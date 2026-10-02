import pytest

from dexkit.hw.grbl_protocol import (
    format_settings,
    jog_command,
    move_command,
    parse_line,
    parse_settings,
    parse_status,
    work_position,
)

SETTINGS_DUMP = """$0=10
$1=25
$10=1
$11=0.010
$20=0
$21=0
$22=0
$100=800.000
$110=2000.000
$120=100.000
$130=300.000
$131=180.000
$132=45.000""".splitlines()


def test_idle_mpos_fs():
    r = parse_status("<Idle|MPos:0.000,0.000,0.000|FS:0,0>")
    assert r.state == "Idle" and r.mpos == (0.0, 0.0, 0.0) and r.feed == 0.0 and r.spindle == 0.0


def test_jog_with_buffer():
    r = parse_status("<Jog|MPos:1.500,-2.000,3.250|Bf:15,128|FS:600,0>")
    assert r.state == "Jog" and r.mpos == (1.5, -2.0, 3.25)
    assert r.planner_free == 15 and r.rx_free == 128 and r.feed == 600


@pytest.mark.parametrize("word", ["Idle", "Run", "Hold:0", "Hold:1", "Jog", "Alarm", "Door:0", "Check", "Home", "Sleep"])
def test_all_state_words(word):
    r = parse_status(f"<{word}|MPos:0.000,0.000,0.000|FS:0,0>")
    assert r.state == word.split(":")[0]
    assert r.known_state
    if ":" in word:
        assert r.substate == word.split(":")[1]


def test_alarm_with_pins_and_wco():
    r = parse_status("<Alarm|MPos:10.000,20.000,-5.000|FS:0,0|Pn:XZ|WCO:10.000,20.000,0.000>")
    assert r.state == "Alarm" and r.pins == "XZ" and r.wco == (10.0, 20.0, 0.0)
    assert work_position(r, None) == (0.0, 0.0, -5.0)


def test_wpos_and_f_field_and_unknown_field():
    r = parse_status("<Run|WPos:1.000,2.000,3.000|F:500|Ov:100,100,100>")
    assert r.wpos == (1.0, 2.0, 3.0) and r.feed == 500 and r.extra["Ov"] == "100,100,100"


def test_grbl_09_status():
    r = parse_status("<Idle,MPos:5.000,6.000,7.000,WPos:1.000,2.000,3.000,Buf:0>")
    assert r.state == "Idle" and r.mpos == (5.0, 6.0, 7.0) and r.wpos == (1.0, 2.0, 3.0)


def test_unknown_state_flagged():
    assert not parse_status("<Weird|MPos:0,0,0>").known_state


def test_response_lines():
    assert parse_line("ok").kind == "ok"
    e = parse_line("error:9")
    assert e.kind == "error" and e.code == 9
    a = parse_line("ALARM:1")
    assert a.kind == "alarm" and a.code == 1
    b = parse_line("Grbl 1.1f ['$' for help]")
    assert b.kind == "banner" and b.text == "1.1f"
    assert parse_line("Grbl 0.9j ['$' for help]").text == "0.9j"
    assert parse_line("[MSG:'$H'|'$X' to unlock]").kind == "message"
    assert parse_line("$130=300.000").setting == (130, 300.0)
    assert parse_line("").kind == "empty"
    assert parse_line("<Idle|MPos:0.000,0.000,0.000|FS:0,0>").kind == "status"


def test_settings_dump_round_trip():
    s = parse_settings(SETTINGS_DUMP)
    assert s[130] == 300.0 and s[22] == 0 and s[11] == pytest.approx(0.01)
    assert parse_settings(format_settings(s)) == s


def test_gcode_formatting():
    assert jog_command(1, 0, -0.5, 600) == "$J=G91 G21 X1.000 Z-0.500 F600"
    assert move_command(10, 20.5, -3, 800) == "G90 G21 G1 X10.000 Y20.500 Z-3.000 F800"


def test_malformed_status_raises():
    with pytest.raises(ValueError):
        parse_status("Idle|MPos:0,0,0")
