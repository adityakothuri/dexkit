"""Interactive gantry calibration: jog, set zero, measure the travel box.

Keys: w a s d (Y+ X- Y- X+), q / e (Z up / down), Shift = 10 mm steps
      z  set zero here (G92 X0 Y0 Z0)       m  mark this position as a box corner
      Enter or x  write gantry.yaml          Esc  quit without writing

The box is the bounding box of the zero point and every marked corner, shrunk by
margin_mm on each face. A warning is printed if it exceeds $130/$131/$132.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from dexkit.config import config_dir, data_dir, dump_yaml, gantry_config_from_dict, load_yaml
from dexkit.control.teleop import JOG_KEYS, KeyEvent, KeySource, ScriptedKeys, TerminalKeys
from dexkit.hw.safety import TravelBox
from dexkit.util import setup_logging

HEADER = """# DexKit gantry configuration (travel box written by dexkit-calibrate-gantry).
# Gantry actions are millimetres in the zeroed frame.
"""

DEFAULT_MOCK_SCRIPT = ",".join(
    ["0.2:z"] + [f"{0.3 + 0.05 * i:.2f}:D" for i in range(20)] + [f"{1.4 + 0.05 * i:.2f}:W" for i in range(12)]
    + [f"{2.1 + 0.05 * i:.2f}:E" for i in range(3)] + ["3.0:m", "3.3:x"]
)


def compute_box(points: list[np.ndarray], margin: float) -> tuple[np.ndarray, np.ndarray]:
    pts = np.array(points, dtype=float)
    lo, hi = pts.min(axis=0) + margin, pts.max(axis=0) - margin
    if np.any(lo >= hi):
        raise ValueError(f"measured box too small for a {margin} mm margin: {lo.tolist()}..{hi.tolist()}")
    return lo, hi


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="output yaml (default config/gantry.yaml; data/mock/gantry.yaml with --mock)")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--scripted", action="store_true", help="run a built-in key script (for --mock)")
    p.add_argument("--script", help="custom key script 't:key,...'")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)

    src = config_dir() / "gantry.yaml"
    out = Path(args.out) if args.out else (data_dir() / "mock" / "gantry.yaml" if args.mock else src)
    raw = load_yaml(out if out.exists() else src)
    cfg = gantry_config_from_dict(raw)

    if args.mock:
        from dexkit.hw.mock import MockGantry

        g = MockGantry(cfg, speed_factor=10.0)
    else:
        from dexkit.hw.grbl_gantry import GrblGantry

        g = GrblGantry(cfg)
    g.connect()
    print(f"GRBL {g.version}")
    for k in sorted(g.settings):
        print(f"  ${k}={g.settings[k]:g}")
    max_travel = np.array([g.settings.get(n, 1000.0) for n in (130, 131, 132)])
    g.box = TravelBox(-max_travel, max_travel)  # don't let the old box limit the measurement

    keys: KeySource
    if args.script or args.scripted:
        keys = ScriptedKeys.parse(args.script or DEFAULT_MOCK_SCRIPT)
    elif sys.stdin.isatty():
        keys = TerminalKeys()
    else:
        print("needs a TTY (or --scripted)")
        sys.exit(2)

    print(__doc__)
    points: list[np.ndarray] = []
    write = False
    try:
        while True:
            evs: list[KeyEvent] = keys.poll()
            done = False
            for ev in evs:
                k = ev.key
                if k == "esc":
                    done = True
                elif k in ("x", "\n"):
                    write, done = True, True
                elif k == "z":
                    g.set_zero()
                    points = [np.zeros(3)]
                    print("zero set")
                elif k == "m":
                    if not g.frame_valid:
                        print("set zero (z) first")
                        continue
                    st = g.wait_idle(timeout=60)
                    points.append(st.xyz.copy())
                    print(f"marked {np.round(st.xyz, 2).tolist()}")
                elif k.lower() in JOG_KEYS:
                    step = cfg.jog_step_big_mm if k.isupper() else cfg.jog_step_mm
                    g.jog(*(v * step for v in JOG_KEYS[k.lower()]))
            if done:
                break
            time.sleep(0.03)
    finally:
        keys.close()

    try:
        if not write:
            print("not written")
            return
        if len(points) < 2:
            print("need the zero point plus at least one marked corner; not written")
            sys.exit(1)
        lo, hi = compute_box(points, cfg.margin_mm)
        extent = hi - lo
        if np.any(extent > max_travel):
            print(f"WARNING: box extent {extent.round(1).tolist()} exceeds $130-$132 {max_travel.tolist()}")
        raw["travel_box"] = {"min": [round(float(v), 2) for v in lo], "max": [round(float(v), 2) for v in hi]}
        raw["calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        gantry_config_from_dict(raw)  # validate
        dump_yaml(raw, out, header=HEADER)
        print(f"travel box min {raw['travel_box']['min']} max {raw['travel_box']['max']}")
        print(f"wrote {out}")
        if not args.mock:
            g.box = TravelBox(lo, hi)
    finally:
        g.close()


if __name__ == "__main__":
    main()
