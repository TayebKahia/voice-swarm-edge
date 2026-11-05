#!/usr/bin/env python3
"""Exp-4 clamp ablation: does the potential field alone keep the vehicles apart? (P14, B-M6)

Exp-4 reports zero collisions, but collisions are counted after the hard separation
clamp, so the zero holds by construction. `results/exp4_spawn.md` shows that 423 of
the clamp's 432 interventions separate vehicles that spawn closer than 0.80 m, before
the controller has acted. This script asks what the field does without the clamp.

Three conditions, each 50 seeds x 3 formations (seed = trial index x 42):

- **clamp-on, protocol spawn**: the published Exp-4 run, reproduced through
  `swarm.simulate.run_trial` unchanged (its activations must match `results/exp4.csv`).
- **clamp-off, protocol spawn**: the same trials with the clamp replaced by a no-op.
- **clamp-off, separated spawn**: the same seeds, but the spawn is redrawn from the same
  N(0, 0.5 m) until no pair starts closer than 0.80 m (rejection sampling). The dispersion
  stays at the protocol's, so this differs from the previous condition only in the close
  pairs. Raising sigma instead would need about 5 m to clear every seed, which is as wide
  as the formation itself and would turn the trial into a long approach.

Per trial: the closest approach over the 60 s, the ticks whose closest pair is under the
0.35 m collision distance, and the ticks under the 0.80 m clamp distance. With the clamp
on, these are measured after the clamp, as Exp-4 reports them.

Because almost every close pair is a spawn pair, each trial is also split at its
*separation tick*: the first tick at which no pair is under 0.80 m. What follows is the
in-flight phase, where the field (not the spawn) decides the spacing. With the clamp on,
the in-flight figures are taken from the state *before* the clamp acts, from the second
tick on (the first tick's clamp is the one that separates the spawn pairs), so they show
what the field let the dynamics do. Medians carry
order-statistic 95% intervals (`eval/stats.py`); trials with a collision are counts with
exact (Clopper-Pearson) 95% intervals. No ANOVA: the simulator is deterministic (P17).

Usage:
    python eval/exp4_clamp_ablation.py
"""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):  # `python eval/exp4_clamp_ablation.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import swarm.simulate as simulate
from eval.stats import order_statistic_ci
from eval.wake_real import clopper_pearson
from swarm.control import CLAMP_DISTANCE, COLLISION_DISTANCE, SHAPES
from swarm.env import NumpyEnv

REPO = Path(__file__).resolve().parent.parent
RECORDED = REPO / "results" / "exp4.csv"
OUT_CSV = REPO / "results" / "exp4_clamp_ablation.csv"
OUT_MD = REPO / "results" / "exp4_clamp_ablation.md"
TRIALS = 50

#: The clamp leaves a corrected pair a hair under `CLAMP_DISTANCE` (its own tolerance is
#: 1e-9 m), so "under 0.80 m" is counted below `CLAMP_DISTANCE - TOL`; one micrometre is
#: far below anything the field or the clamp could mean.
TOL = 1e-6

CONDITIONS = (
    ("clamp-on, protocol spawn", True, False),
    ("clamp-off, protocol spawn", False, False),
    ("clamp-off, separated spawn", False, True),
)


def _closest(positions: np.ndarray) -> float:
    distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


class SeparatedSpawnEnv(NumpyEnv):
    """The protocol spawn, redrawn from the same seeded generator until no pair is close.

    Each redraw is a full N(0, sigma) draw about the centroid with the floor applied, as in
    `NumpyEnv.reset`, so the accepted spawn is the protocol distribution conditioned on
    every pair being at least `CLAMP_DISTANCE` apart.
    """

    draws: int = 0

    def reset(self, seed: int):
        rng = np.random.default_rng(seed)
        self.draws = 0
        while True:
            self.draws += 1
            positions = self.centroid + rng.normal(0.0, self.spawn_sigma, size=(self.n, 3))
            positions[:, 2] = np.maximum(positions[:, 2], self.limits.min_altitude)
            if _closest(positions) >= CLAMP_DISTANCE:
                break
        self._positions = positions
        self._velocities = np.zeros((self.n, 3))
        return self.positions, self.velocities


def run_one(spec: simulate.TrialSpec, clamp: bool, separated: bool) -> dict:
    """One trial with the clamp kept or replaced; per-tick closest distances recorded."""
    pre: list[float] = []      # closest pair the integrator produced, before any clamp
    post: list[float] = []     # closest pair in the recorded state
    original = simulate.separation_clamp

    def wrapper(positions, velocities, *args, **kwargs):
        pre.append(_closest(positions))
        if clamp:
            result = original(positions, velocities, *args, **kwargs)
        else:
            result = (np.asarray(positions, dtype=float).copy(),
                      np.asarray(velocities, dtype=float).copy(), 0)
        post.append(_closest(result[0]))
        return result

    if separated:
        env = SeparatedSpawnEnv(n=spec.n, dt=spec.dt, spawn_sigma=spec.spawn_sigma)
    else:
        env = NumpyEnv(n=spec.n, dt=spec.dt, spawn_sigma=spec.spawn_sigma)
    spawn_closest = _closest(env.reset(spec.seed)[0])
    draws = getattr(env, "draws", 1)

    simulate.separation_clamp = wrapper
    try:
        # run_trial resets the env itself with the same seed, so the spawn is identical.
        result = simulate.run_trial(spec, env=env)
    finally:
        simulate.separation_clamp = original

    post_arr, pre_arr = np.array(post), np.array(pre)
    separated_at = np.flatnonzero(post_arr >= CLAMP_DISTANCE - TOL)
    sep_tick = int(separated_at[0]) if separated_at.size else len(post_arr)
    # In flight: with the clamp off, the recorded state from the separation tick on;
    # with it on, the pre-clamp state from the second tick on.
    flight = pre_arr[1:] if clamp else post_arr[sep_tick:]
    assert int((post_arr < COLLISION_DISTANCE).sum()) == result.collisions
    return {
        "shape": spec.shape, "trial": spec.index, "seed": spec.seed,
        "spawn_draws": draws,
        "spawn_closest": round(spawn_closest, 4),
        "closest_approach": round(float(post_arr.min()), 4),
        "closest_before_clamp": round(min(pre), 4),
        "ticks_under_collision": int((post_arr < COLLISION_DISTANCE).sum()),
        "ticks_under_clamp": int((post_arr < CLAMP_DISTANCE - TOL).sum()),
        "separation_tick": sep_tick,
        "flight_closest": round(float(flight.min()), 4) if flight.size else "",
        "flight_ticks_under_clamp": int((flight < CLAMP_DISTANCE - TOL).sum()),
        "flight_ticks_under_collision": int((flight < COLLISION_DISTANCE).sum()),
        "clamp_activations": result.clamp_activations,
        "formation_accuracy": round(result.formation_accuracy, 6),
    }


def _median_ci(values: list[float], fmt: Callable[[float], str]) -> str:
    low, high = order_statistic_ci(values, 50)
    return f"{fmt(float(np.median(values)))} [{fmt(low)}, {fmt(high)}]"


def _count_ci(k: int, n: int) -> str:
    low, high = clopper_pearson(k, n)
    return f"{k}/{n} [{low:.2f}, {high:.2f}]"


def run() -> int:
    with RECORDED.open(encoding="utf-8") as handle:
        recorded = {(r["shape"], int(r["trial"])): int(r["clamp_activations"])
                    for r in csv.DictReader(handle)}

    rows = []
    for label, clamp, separated in CONDITIONS:
        for shape, index in itertools.product(SHAPES, range(TRIALS)):
            row = run_one(simulate.TrialSpec(shape=shape, index=index), clamp, separated)
            rows.append({"condition": label, **row})
        print(f"  {label}: done")

    mismatches = [(r["shape"], r["trial"], r["clamp_activations"],
                   recorded[(r["shape"], r["trial"])])
                  for r in rows if r["condition"] == CONDITIONS[0][0]
                  and r["clamp_activations"] != recorded[(r["shape"], r["trial"])]]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    m = lambda v: f"{v:.3f}"
    i = lambda v: "inf" if v == float("inf") else f"{v:.0f}"
    lines = [
        "<!-- Generated by eval/exp4_clamp_ablation.py. Do not edit by hand. -->",
        "",
        "### Exp-4 clamp ablation: the potential field with and without the separation clamp",
        "",
        f"{TRIALS} seeds x {len(SHAPES)} formations per condition, 60 s at 50 Hz "
        f"(3000 ticks per trial), seed = trial index x 42, kinematic backend. "
        f"Collision distance {COLLISION_DISTANCE:.2f} m; clamp distance {CLAMP_DISTANCE:.2f} m. "
        "Separated spawn: the protocol N(0, 0.5 m) spawn redrawn from the same seeded "
        f"generator until no pair starts closer than {CLAMP_DISTANCE:.2f} m.",
        "",
        "Medians are per trial with order-statistic 95% intervals; collision trials are "
        "counts with exact (Clopper-Pearson) 95% intervals. With the clamp on, distances "
        "and ticks are measured after the clamp, as Exp-4 reports them.",
        "",
        "| Condition | Formation | n | Spawn closest, median [95% CI] m | "
        "Closest approach, median [95% CI] m | Min | Trials with a collision [95% CI] | "
        f"Ticks < {COLLISION_DISTANCE:.2f} m, total | Ticks < {CLAMP_DISTANCE:.2f} m, "
        "median [95% CI] | Ticks < 0.80 m, total |",
        "| :--- | :--- | ---: | :--- | :--- | ---: | :--- | ---: | :--- | ---: |",
    ]
    for label, _, _ in CONDITIONS:
        for shape in (*SHAPES, "all"):
            group = [r for r in rows if r["condition"] == label
                     and (shape == "all" or r["shape"] == shape)]
            n = len(group)
            collided = sum(r["ticks_under_collision"] > 0 for r in group)
            lines.append(
                f"| {label} | {shape} | {n} "
                f"| {_median_ci([r['spawn_closest'] for r in group], m)} "
                f"| {_median_ci([r['closest_approach'] for r in group], m)} "
                f"| {min(r['closest_approach'] for r in group):.3f} "
                f"| {_count_ci(collided, n)} "
                f"| {sum(r['ticks_under_collision'] for r in group)} "
                f"| {_median_ci([r['ticks_under_clamp'] for r in group], i)} "
                f"| {sum(r['ticks_under_clamp'] for r in group)} |")

    lines += [
        "",
        "In flight, after each trial's separation tick (clamp-on: the pre-clamp state from "
        "the second tick on). Separation tick: the first tick with no pair under "
        f"{CLAMP_DISTANCE:.2f} m (50 Hz, so 50 ticks = 1 s).",
        "",
        "| Condition | Formation | n | Separation tick, median [95% CI] | max "
        "| In-flight closest, median [95% CI] m | Min "
        f"| Trials back under {CLAMP_DISTANCE:.2f} m [95% CI] "
        f"| Ticks < {CLAMP_DISTANCE:.2f} m, total "
        f"| Trials under {COLLISION_DISTANCE:.2f} m [95% CI] |",
        "| :--- | :--- | ---: | :--- | ---: | :--- | ---: | :--- | ---: | :--- |",
    ]
    for label, _, _ in CONDITIONS:
        for shape in (*SHAPES, "all"):
            group = [r for r in rows if r["condition"] == label
                     and (shape == "all" or r["shape"] == shape)]
            n = len(group)
            back = sum(r["flight_ticks_under_clamp"] > 0 for r in group)
            hit = sum(r["flight_ticks_under_collision"] > 0 for r in group)
            lines.append(
                f"| {label} | {shape} | {n} "
                f"| {_median_ci([r['separation_tick'] for r in group], i)} "
                f"| {max(r['separation_tick'] for r in group)} "
                f"| {_median_ci([r['flight_closest'] for r in group], m)} "
                f"| {min(r['flight_closest'] for r in group):.3f} "
                f"| {_count_ci(back, n)} "
                f"| {sum(r['flight_ticks_under_clamp'] for r in group)} "
                f"| {_count_ci(hit, n)} |")

    on = [r for r in rows if r["condition"] == CONDITIONS[0][0]]
    sep = [r for r in rows if r["condition"] == CONDITIONS[2][0]]
    lines += [
        "",
        "The \"all\" rows pool the three formations, which share their 50 spawns; the "
        "per-formation rows are the independent samples.",
        "",
        f"Clamp-on, closest pair the dynamics produced before the clamp acted: median "
        f"{_median_ci([r['closest_before_clamp'] for r in on], m)} m, minimum "
        f"{min(r['closest_before_clamp'] for r in on):.3f} m; clamp activations "
        f"{sum(r['clamp_activations'] for r in on)}.",
        f"Separated spawn: {int(np.median([r['spawn_draws'] for r in sep]))} draws per seed "
        f"at the median, at most {max(r['spawn_draws'] for r in sep)}.",
        "",
        "Clamp-on activations reproduce `results/exp4.csv`." if not mismatches
        else f"MISMATCH against results/exp4.csv: {mismatches}",
    ]
    text = "\n".join(lines) + "\n"
    OUT_MD.write_text(text, encoding="utf-8")
    print(text)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(run())
