"""The offset between the Raspberry Pi's monotonic clock and the workstation's (peer-review item P11).

A closed-loop latency starts on the Pi (utterance onset, in the capture's stamps) and ends on the
workstation (the vehicles' arrest, in the simulation's ticks). Neither machine's `time.monotonic()`
means anything to the other, and `systemd-timesyncd` disciplines only the wall clocks, and only
while the internet is reachable, which the offline runs remove. So the two monotonic clocks are
compared directly, over the same Wi-Fi link the bus uses, NTP-style:

    Pi sends t1 (Pi clock)  ->  workstation stamps t2 (its clock) and echoes  ->  Pi stamps t3

    offset = t2 - (t1 + t3) / 2        workstation clock minus Pi clock
    error  <= (t3 - t1) / 2            whatever the split of the round trip between directions

`serve()` is the workstation's echo; `demo/workstation.py --clock-port` runs it in a thread.
`python demo/clock.py probe HOST` runs on the Pi alongside `runtime/main.py`, one probe a second,
appending `{"t1", "t2", "t3"}` lines to a JSONL file. `fit()` keeps the probes with the smallest
round trips and fits a line through their offsets, which absorbs the two crystals' drift over a run;
the half round trip of the probes it kept is the stated bound on the offset's error.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np

CLOCK_PORT = 8767

__all__ = ["CLOCK_PORT", "ClockFit", "LocalClock", "serve", "probe", "fit"]


def serve(port: int = CLOCK_PORT, bind: str = "0.0.0.0", stop: threading.Event | None = None) -> None:
    """Echo each probe with this machine's monotonic stamp, until `stop` is set."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind, port))
    sock.settimeout(0.2)
    try:
        while stop is None or not stop.is_set():
            try:
                data, addr = sock.recvfrom(512)
            except socket.timeout:
                continue
            t2 = time.monotonic()
            try:
                t1 = float(json.loads(data)["t1"])
            except (ValueError, KeyError, TypeError):
                continue
            sock.sendto(json.dumps({"t1": t1, "t2": t2}).encode(), addr)
    finally:
        sock.close()


def serve_in_thread(port: int = CLOCK_PORT, bind: str = "0.0.0.0") -> threading.Event:
    stop = threading.Event()
    threading.Thread(target=serve, args=(port, bind, stop), name="clock-echo", daemon=True).start()
    return stop


def probe(host: str, out: Path, *, port: int = CLOCK_PORT, interval: float = 1.0,
          seconds: float | None = None, timeout: float = 0.5) -> int:
    """Probe `host` every `interval` s, appending one JSON line per answered probe to `out`."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    deadline = None if seconds is None else time.monotonic() + seconds
    with open(out, "a", encoding="utf-8") as fh:
        try:
            while deadline is None or time.monotonic() < deadline:
                t1 = time.monotonic()
                sock.sendto(json.dumps({"t1": t1}).encode(), (host, port))
                try:
                    while True:
                        data, _ = sock.recvfrom(512)
                        t3 = time.monotonic()
                        reply = json.loads(data)
                        if float(reply["t1"]) == t1:          # not a late answer to an earlier probe
                            break
                    fh.write(json.dumps({"t1": t1, "t2": float(reply["t2"]), "t3": t3}) + "\n")
                    fh.flush()
                    n += 1
                except socket.timeout:
                    pass
                time.sleep(max(0.0, interval - (time.monotonic() - t1)))
        except KeyboardInterrupt:
            pass
    sock.close()
    return n


class ClockFit:
    """offset(t_pi) = slope * t_pi + intercept: workstation clock minus Pi clock."""

    def __init__(self, slope: float, intercept: float, bound_s: float, n_used: int, n_total: int,
                 residual_s: float) -> None:
        self.slope, self.intercept = slope, intercept
        self.bound_s, self.n_used, self.n_total, self.residual_s = bound_s, n_used, n_total, residual_s

    def offset(self, t_pi: float) -> float:
        return self.slope * t_pi + self.intercept

    def to_workstation(self, t_pi: float) -> float:
        return t_pi + self.offset(t_pi)

    def as_dict(self) -> dict:
        return {"slope": self.slope, "intercept": self.intercept, "bound_ms": self.bound_s * 1000,
                "n_used": self.n_used, "n_total": self.n_total, "residual_ms": self.residual_s * 1000}


def fit(samples: Sequence[dict], keep: float = 0.2) -> ClockFit:
    """Fit the offset through the `keep` fraction of probes with the smallest round trip.

    A probe's offset can be wrong by at most half its round trip, and the smallest round trips
    are the least queued; the bound reported is the largest half round trip among those kept.
    """
    t1 = np.array([s["t1"] for s in samples], dtype=float)
    t2 = np.array([s["t2"] for s in samples], dtype=float)
    t3 = np.array([s["t3"] for s in samples], dtype=float)
    if t1.size < 2:
        raise ValueError(f"need at least two probes, got {t1.size}")
    rtt = t3 - t1
    offset = t2 - (t1 + t3) / 2
    k = max(2, int(np.ceil(keep * t1.size)))
    idx = np.argsort(rtt)[:k]
    mid = (t1[idx] + t3[idx]) / 2
    if np.ptp(mid) > 0:
        slope, intercept = np.polyfit(mid, offset[idx], 1)
    else:
        slope, intercept = 0.0, float(np.mean(offset[idx]))
    resid = offset[idx] - (slope * mid + intercept)
    return ClockFit(float(slope), float(intercept), float(rtt[idx].max() / 2), int(k), int(t1.size),
                    float(np.abs(resid).max()))


class LocalClock:
    """The offset at one instant, from the probes around it rather than one line through the run.

    Two crystals' drift is not linear over tens of minutes (it follows temperature), so a line through
    a whole run can miss a given instant by more than its probes' round trips. Here the offset at `t`
    is the mean of the `k` smallest-round-trip probes within `window` seconds of it, and its bound is
    the largest half round trip among them plus their spread.
    """

    def __init__(self, samples: Sequence[dict], window: float = 60.0, k: int = 3) -> None:
        self.mid = np.array([(s["t1"] + s["t3"]) / 2 for s in samples], dtype=float)
        self.rtt = np.array([s["t3"] - s["t1"] for s in samples], dtype=float)
        self.off = np.array([s["t2"] - (s["t1"] + s["t3"]) / 2 for s in samples], dtype=float)
        self.window, self.k = window, k

    def at(self, t_pi: float) -> tuple[float, float]:
        near = np.flatnonzero(np.abs(self.mid - t_pi) <= self.window)
        if near.size == 0:
            near = np.argsort(np.abs(self.mid - t_pi))[: self.k]
        best = near[np.argsort(self.rtt[near])[: self.k]]
        offset = float(self.off[best].mean())
        bound = float(self.rtt[best].max() / 2 + np.ptp(self.off[best]) / 2)
        return offset, bound

    def to_workstation(self, t_pi: float) -> float:
        return t_pi + self.at(t_pi)[0]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="echo probes (the workstation)")
    s.add_argument("--port", type=int, default=CLOCK_PORT)
    p = sub.add_parser("probe", help="probe the workstation (the Pi)")
    p.add_argument("host")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=CLOCK_PORT)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--seconds", type=float, default=None)
    f = sub.add_parser("fit", help="fit a probe file")
    f.add_argument("path", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "serve":
        serve(args.port)
    elif args.cmd == "probe":
        n = probe(args.host, args.out, port=args.port, interval=args.interval, seconds=args.seconds)
        print(f"{n} probes answered", file=sys.stderr)
    else:
        print(json.dumps(fit(load(args.path)).as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
