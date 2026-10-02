"""SPACE = emergency stop while the hand (or gantry) is moving, in any command.

`with SpaceWatch(estop):` puts the terminal into raw-ish (cbreak) mode and watches the
keyboard from a background thread for the duration of a motion. Space requests an
e-stop; the control loop's next `estop.check()` performs the hardware stop (torque off,
gantry hold) and raises, so the motion ends and nothing else runs until the command is
restarted. Without a TTY (scripts, --mock runs under make) it does nothing.
"""

from __future__ import annotations

import logging
import os
import select
import sys
import threading
from collections.abc import Callable

from dexkit.hw.safety import EStop

log = logging.getLogger(__name__)

STOP_KEYS = (b" ",)


class SpaceWatch:
    def __init__(self, estop: EStop, reader: Callable[[float], bytes | None] | None = None,
                 enabled: bool | None = None) -> None:
        self.estop = estop
        self._reader = reader  # tests inject one; default reads the terminal
        self._enabled = enabled if enabled is not None else (reader is not None or sys.stdin.isatty())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._saved = None
        self._fd: int | None = None

    # -- terminal plumbing
    def _enter_cbreak(self) -> None:
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def _leave_cbreak(self) -> None:
        if self._fd is not None and self._saved is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        self._fd, self._saved = None, None

    def _read_terminal(self, timeout: float) -> bytes | None:
        assert self._fd is not None
        r, _, _ = select.select([self._fd], [], [], timeout)
        return os.read(self._fd, 1) if r else None

    # -- thread
    def _run(self) -> None:
        read = self._reader or self._read_terminal
        while not self._stop.is_set():
            try:
                key = read(0.05)
            except Exception as e:  # noqa: BLE001 - a watcher must never take the control loop down
                log.debug("key watcher: %s", e)
                return
            if key in STOP_KEYS:
                self.estop.request("operator pressed SPACE")
                return

    def __enter__(self) -> SpaceWatch:
        if not self._enabled:
            return self
        if self._reader is None:
            try:
                self._enter_cbreak()
            except Exception as e:  # noqa: BLE001 - no terminal control: run without the watcher
                log.debug("no cbreak terminal (%s); SPACE e-stop unavailable here", e)
                self._enabled = False
                return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="space-estop", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if not self._enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._reader is None:
            self._leave_cbreak()
