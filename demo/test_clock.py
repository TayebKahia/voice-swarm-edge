"""demo/clock.py: the offset fit recovers a known offset and drift, and the echo answers."""

import threading
from pathlib import Path

import numpy as np

from demo.clock import fit, load, probe, serve


def test_fit_recovers_offset_and_drift():
    rng = np.random.default_rng(0)
    t1 = np.arange(0.0, 600.0, 1.0) + 1000.0
    up = 0.002 + rng.exponential(0.004, t1.size)       # queueing only ever adds delay
    down = 0.002 + rng.exponential(0.004, t1.size)
    true = 5.0 + 20e-6 * t1                              # 5 s offset, 20 ppm drift
    t2 = t1 + up + true
    t3 = t1 + up + down
    samples = [{"t1": a, "t2": b, "t3": c} for a, b, c in zip(t1, t2, t3)]
    f = fit(samples)
    err = np.abs([f.offset(t) - (5.0 + 20e-6 * t) for t in t1])
    assert err.max() < f.bound_s + 1e-9
    assert f.bound_s < 0.01


def test_probe_against_local_echo(tmp_path: Path):
    stop = threading.Event()
    threading.Thread(target=serve, args=(18767, "127.0.0.1", stop), daemon=True).start()
    n = probe("127.0.0.1", tmp_path / "p.jsonl", port=18767, interval=0.05, seconds=0.5)
    stop.set()
    assert n >= 5
    samples = load(tmp_path / "p.jsonl")
    assert abs(fit(samples).offset(samples[0]["t1"])) < 0.01    # one machine: no offset
