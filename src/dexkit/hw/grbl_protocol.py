"""Pure parsers for GRBL 1.1 (and 0.9 status lines). No I/O here."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

STATE_WORDS = ("Idle", "Run", "Hold", "Jog", "Alarm", "Door", "Check", "Home", "Sleep")

RT_STATUS = b"?"
RT_FEED_HOLD = b"!"
RT_RESUME = b"~"
RT_SOFT_RESET = b"\x18"
RT_JOG_CANCEL = b"\x85"

BANNER_RE = re.compile(r"^Grbl\s+(\d+\.\d+\w*)")
SETTING_RE = re.compile(r"^\$(\d+)=([-+]?\d*\.?\d+)")


@dataclass
class StatusReport:
    state: str
    substate: str | None = None
    mpos: tuple[float, float, float] | None = None
    wpos: tuple[float, float, float] | None = None
    wco: tuple[float, float, float] | None = None
    feed: float | None = None
    spindle: float | None = None
    planner_free: int | None = None
    rx_free: int | None = None
    pins: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def known_state(self) -> bool:
        return self.state in STATE_WORDS


def _xyz(text: str) -> tuple[float, float, float]:
    vals = [float(v) for v in text.split(",")]
    if len(vals) < 3:
        raise ValueError(f"expected 3 axes in '{text}'")
    return vals[0], vals[1], vals[2]


def parse_status(line: str) -> StatusReport:
    """Parse '<Idle|MPos:1.000,2.000,3.000|FS:0,0>' (1.1) or '<Idle,MPos:...,WPos:...>' (0.9)."""
    line = line.strip()
    if not (line.startswith("<") and line.endswith(">")):
        raise ValueError(f"not a status report: {line!r}")
    body = line[1:-1]
    if "|" not in body and re.match(r"^\w+,", body):
        return _parse_status_09(body)
    return _parse_status_11(body)


def _parse_status_11(body: str) -> StatusReport:
    fields = body.split("|")
    state, _, sub = fields[0].partition(":")
    r = StatusReport(state=state, substate=sub or None)
    for f in fields[1:]:
        key, _, val = f.partition(":")
        if key == "MPos":
            r.mpos = _xyz(val)
        elif key == "WPos":
            r.wpos = _xyz(val)
        elif key == "WCO":
            r.wco = _xyz(val)
        elif key == "FS":
            parts = val.split(",")
            r.feed = float(parts[0])
            r.spindle = float(parts[1]) if len(parts) > 1 else None
        elif key == "F":
            r.feed = float(val)
        elif key == "Bf":
            parts = val.split(",")
            r.planner_free = int(parts[0])
            r.rx_free = int(parts[1]) if len(parts) > 1 else None
        elif key == "Pn":
            r.pins = val
        else:
            r.extra[key] = val
    return r


def _parse_status_09(body: str) -> StatusReport:
    state, rest = body.split(",", 1)
    r = StatusReport(state=state)
    for key in ("MPos", "WPos"):
        m = re.search(key + r":([-\d.]+),([-\d.]+),([-\d.]+)", rest)
        if m:
            setattr(r, key.lower(), (float(m[1]), float(m[2]), float(m[3])))
    m = re.search(r"Buf:(\d+)", rest)
    if m:
        r.extra["Buf"] = m[1]
    return r


def work_position(r: StatusReport, last_wco: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    """WPos = MPos - WCO (GRBL 1.1 reports one of MPos/WPos depending on $10)."""
    if r.wpos is not None:
        return r.wpos
    wco = r.wco or last_wco
    if r.mpos is not None and wco is not None:
        return tuple(m - w for m, w in zip(r.mpos, wco, strict=True))  # type: ignore[return-value]
    return None


def machine_position(r: StatusReport, last_wco: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if r.mpos is not None:
        return r.mpos
    wco = r.wco or last_wco
    if r.wpos is not None and wco is not None:
        return tuple(w + o for w, o in zip(r.wpos, wco, strict=True))  # type: ignore[return-value]
    return None


@dataclass
class Response:
    kind: str                 # ok, error, alarm, status, banner, setting, message, feedback, unknown, empty
    code: int | None = None
    text: str = ""
    status: StatusReport | None = None
    setting: tuple[int, float] | None = None


def parse_line(line: str) -> Response:
    s = line.strip()
    if not s:
        return Response("empty")
    if s == "ok":
        return Response("ok")
    if s.startswith("error:"):
        try:
            return Response("error", code=int(s[6:]), text=s)
        except ValueError:
            return Response("error", text=s)
    if s.startswith("ALARM:"):
        try:
            return Response("alarm", code=int(s[6:]), text=s)
        except ValueError:
            return Response("alarm", text=s)
    if s.startswith("<"):
        return Response("status", text=s, status=parse_status(s))
    m = BANNER_RE.match(s)
    if m:
        return Response("banner", text=m[1])
    m = SETTING_RE.match(s)
    if m:
        return Response("setting", setting=(int(m[1]), float(m[2])), text=s)
    if s.startswith("[MSG:"):
        return Response("message", text=s[5:-1] if s.endswith("]") else s[5:])
    if s.startswith("["):
        return Response("feedback", text=s)
    return Response("unknown", text=s)


def parse_settings(lines: list[str]) -> dict[int, float]:
    out: dict[int, float] = {}
    for line in lines:
        m = SETTING_RE.match(line.strip())
        if m:
            out[int(m[1])] = float(m[2])
    return out


def format_settings(settings: dict[int, float]) -> list[str]:
    return [f"${k}={int(v) if float(v).is_integer() else v}" for k, v in sorted(settings.items())]


def fmt(v: float) -> str:
    return f"{v:.3f}"


def jog_command(dx: float, dy: float, dz: float, feed: float) -> str:
    parts = ["$J=G91 G21"]
    for axis, d in (("X", dx), ("Y", dy), ("Z", dz)):
        if d:
            parts.append(f"{axis}{fmt(d)}")
    parts.append(f"F{feed:.0f}")
    return " ".join(parts)


def move_command(x: float, y: float, z: float, feed: float) -> str:
    return f"G90 G21 G1 X{fmt(x)} Y{fmt(y)} Z{fmt(z)} F{feed:.0f}"
