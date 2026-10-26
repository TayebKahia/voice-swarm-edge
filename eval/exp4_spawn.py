#!/usr/bin/env python3
"""When the separation clamp acts in Exp-4: at the spawn, or while the swarm flies.

Exp-4 counts every pair the clamp finds closer than `CLAMP_DISTANCE` (0.80 m). Each trial
spawns five vehicles from N(0, 0.5 m) about one centroid, so two of them often start
closer than that, and the clamp separates them on the first tick, before the
controller has moved anyone. Such an intervention says nothing about the potential
field. This script re-runs every trial through `swarm.simulate.run_trial` unchanged,
records the clamp's activations tick by tick, and splits them into the first tick and
the rest. It also counts the close pairs in each spawn directly, and checks that the
totals reproduce `results/exp4.csv`.

Usage:
    python eval/exp4_spawn.py
"""

from __future__ import annotations

import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):  # `python eval/exp4_spawn.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import swarm.simulate as simulate
from swarm.control import CLAMP_DISTANCE, SHAPES
from swarm.env import NumpyEnv

REPO = Path(__file__).resolve().parent.parent
RECORDED = REPO / "results" / "exp4.csv"
OUT_MD = REPO / "results" / "exp4_spawn.md"
TRIALS = 50


def close_pairs_at_spawn(spec: simulate.TrialSpec) -> int:
    env = NumpyEnv(n=spec.n, dt=spec.dt, spawn_sigma=spec.spawn_sigma)
    positions, _ = env.reset(spec.seed)
    return sum(np.linalg.norm(positions[i] - positions[j]) < CLAMP_DISTANCE
               for i, j in itertools.combinations(range(spec.n), 2))


def per_tick_activations(spec: simulate.TrialSpec) -> list[int]:
    """Run the published trial with the clamp wrapped, so every tick's count is kept."""
    ticks: list[int] = []
    original = simulate.separation_clamp

    def recording(*args, **kwargs):
        result = original(*args, **kwargs)
        ticks.append(result[2])
        return result

    simulate.separation_clamp = recording
    try:
        result = simulate.run_trial(spec)
    finally:
        simulate.separation_clamp = original
    assert sum(ticks) == result.clamp_activations
    return ticks


def run() -> int:
    with RECORDED.open(encoding="utf-8") as handle:
        recorded = {(r["shape"], int(r["trial"])): int(r["clamp_activations"])
                    for r in csv.DictReader(handle)}
    totals = defaultdict(lambda: {"trials": 0, "with_clamp": 0, "spawn_pairs": 0,
                                  "first_tick": 0, "later": 0, "trials_later": 0})
    mismatches = []
    seeds_with_close_pair = set()
    for shape, index in itertools.product(SHAPES, range(TRIALS)):
        spec = simulate.TrialSpec(shape=shape, index=index)
        ticks = per_tick_activations(spec)
        spawn = close_pairs_at_spawn(spec)
        if spawn:
            seeds_with_close_pair.add(spec.seed)
        if sum(ticks) != recorded[(shape, index)]:
            mismatches.append((shape, index, sum(ticks), recorded[(shape, index)]))
        t = totals[shape]
        t["trials"] += 1
        t["with_clamp"] += sum(ticks) > 0
        t["spawn_pairs"] += spawn
        t["first_tick"] += ticks[0]
        t["later"] += sum(ticks[1:])
        t["trials_later"] += sum(ticks[1:]) > 0

    lines = ["### Exp-4: separation-clamp activations at the spawn and in flight", "",
             f"{TRIALS} trials per formation, seeds = trial index x 42 (the same 50 spawns for",
             f"every formation). Close pair: two vehicles closer than {CLAMP_DISTANCE:.2f} m.",
             f"{len(seeds_with_close_pair)} of {TRIALS} spawns contain at least one close pair.",
             "", "| Formation | Trials | Trials with an activation | Close pairs at spawn | "
             "Activations, first tick | Activations, later ticks | Trials with a later "
             "activation |", "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for shape in SHAPES:
        t = totals[shape]
        lines.append(f"| {shape} | {t['trials']} | {t['with_clamp']} | {t['spawn_pairs']} | "
                     f"{t['first_tick']} | {t['later']} | {t['trials_later']} |")
    all_first = sum(t["first_tick"] for t in totals.values())
    all_later = sum(t["later"] for t in totals.values())
    lines += ["", f"Total: {all_first + all_later} activations, {all_first} on the first tick "
              f"and {all_later} afterwards.",
              "Totals reproduce `results/exp4.csv`." if not mismatches
              else f"MISMATCH against results/exp4.csv: {mismatches}"]
    text = "\n".join(lines) + "\n"
    OUT_MD.write_text(text, encoding="utf-8")
    print(text)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(run())
