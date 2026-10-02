"""Small shared helpers: logging setup and a fixed-rate loop timer."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        sys.excepthook = _friendly_excepthook


def _friendly_excepthook(etype: type[BaseException], value: BaseException, tb: object) -> None:
    """One readable line instead of a traceback for expected hardware/config problems (-v shows all)."""
    if issubclass(etype, KeyboardInterrupt):
        print("\nstopped (Ctrl+C)", file=sys.stderr)
        return
    if issubclass(etype, (OSError, RuntimeError, ValueError, TimeoutError)):
        print(f"\nerror: {value}\n(run the same command with -v for full details)", file=sys.stderr)
        return
    sys.__excepthook__(etype, value, tb)  # type: ignore[arg-type]


class Rate:
    """Sleep so that successive `sleep()` calls are spaced 1/hz apart."""

    def __init__(
        self,
        hz: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if hz <= 0:
            raise ValueError("rate must be positive")
        self.period = 1.0 / hz
        self._clock = clock
        self._sleep = sleep
        self._next = clock() + self.period
        self.last_overrun = 0.0

    def sleep(self) -> None:
        now = self._clock()
        remaining = self._next - now
        if remaining > 0:
            self._sleep(remaining)
            self.last_overrun = 0.0
        else:
            self.last_overrun = -remaining
        self._next = max(self._next + self.period, self._clock())
