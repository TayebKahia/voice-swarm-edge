#!/usr/bin/env python3
"""Intervals for the latency percentiles, and the reflex results with the take as the unit.

**Percentile intervals.** Every latency verdict in the articles compares a p95 with a
budget. A p95 from 60 trials rests on the three slowest, so a margin of tens of
milliseconds is not resolved by the point estimate alone. Each percentile here carries
a distribution-free 95% interval from order statistics (`eval.stats.order_statistic_ci`);
an upper bound the sample cannot support is printed as "--".

**The reflex take.** Exp-2 replays each of the 40 recorded keyword takes twice per
condition. Replaying the same audio through the same deterministic spotter reproduces
the detection frame, so the second pass is a determinism check, not a second
observation. The reflex rows are therefore recomputed on the first pass of each take,
the replay difference is reported, and detection, cancellation and class confusion are
given as counts over takes with exact (Clopper-Pearson) intervals.

Sources: `results/exp1_cooled.csv` and `results/exp1.csv` (language-model timing, master
article), `results/exp2_raw/*.jsonl` (parse and reflex paths, ingenieur article).

Usage:
    python eval/latency_ci.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):  # `python eval/latency_ci.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import binomtest

from eval.exp2 import algorithmic_ms, first_own_event
from eval.stats import nearest_rank, order_statistic_ci

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT_MD = RESULTS / "latency_ci.md"

EXP1 = {"cooled": RESULTS / "exp1_cooled.csv", "uncooled": RESULTS / "exp1.csv"}
EXP2 = RESULTS / "exp2_raw"
OTHER = {"swarm_hold": "abort", "swarm_abort": "hover"}


def _fmt(value: float) -> str:
    return "--" if math.isinf(value) or math.isnan(value) else f"{value:,.0f}"


def _row(label: str, values: list[float]) -> str:
    values = [v for v in values if v is not None]
    low, high = order_statistic_ci(values, 95)
    p_low, p_high = order_statistic_ci(values, 50)
    return (f"| {label} | {len(values)} | {_fmt(nearest_rank(values, 50))} "
            f"[{_fmt(p_low)}, {_fmt(p_high)}] | {_fmt(nearest_rank(values, 95))} "
            f"[{_fmt(low)}, {_fmt(high)}] |")


HEADER = ["| Measure | n | p50 [95% CI] (ms) | p95 [95% CI] (ms) |",
          "| :--- | ---: | ---: | ---: |"]


def exp1_section() -> list[str]:
    lines = ["### Language-model time per command, Q4_K_M (master article, Exp-1)", "",
             "Scored trials only (warm-up excluded). Prompt cache on, as in the run of record."]
    for run, path in EXP1.items():
        trials: dict[str, list[dict]] = defaultdict(list)
        with path.open(encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                if not r["config"].endswith("-warmup"):
                    trials[r["config"]].append(r)
        lines += ["", f"#### {run}", "", *HEADER]
        for config, rows in sorted(trials.items()):
            prefill = [float(r["slm_prefill_ms"]) for r in rows]
            decode = [float(r["slm_decode_ms"]) for r in rows]
            lines += [_row(f"{config} prefill + decode", [p + d for p, d in zip(prefill, decode)]),
                      _row(f"{config} prefill", prefill), _row(f"{config} decode", decode)]
    return lines


def _jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (EXP2 / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def parse_section() -> list[str]:
    segments = [s for r in _jsonl("branch_b.jsonl") for s in r["segments"]]
    measures = [("Endpointing wait", "vad_wait_ms"), ("Speech recognition", "stt_ms"),
                ("Prefill (no cache)", "prefill_ms"), ("Decode", "decode_ms"),
                ("Validation and dispatch", "validate_fsm_ms"),
                ("End to end, after endpointing", "e2e_t0_ms"),
                ("End to end, including the endpointing wait", "e2e_speech_end_ms")]
    return ["", "### Parse path on the 200 golden recordings (ingenieur article, Exp-2)", "",
            "One sample per endpointed segment.", "", *HEADER,
            *[_row(label, [s.get(key) for s in segments]) for label, key in measures]]


def _ci(successes: int, trials: int) -> str:
    interval = binomtest(successes, trials).proportion_ci(0.95, method="exact")
    return f"{successes}/{trials} [{interval.low:.3f}, {interval.high:.3f}]"


def reflex_section() -> list[str]:
    lines = ["", "### Reflex path with the take as the unit (ingenieur article, Exp-2)", "",
             "First pass of each take. Replay difference: largest absolute difference in",
             "latency from offset between the two passes of a take."]
    for condition in ("idle", "loaded"):
        rows = _jsonl(f"{condition}.jsonl")
        by_take: dict[str, dict[int, dict]] = defaultdict(dict)
        for r in rows:
            by_take[r["take"]][r["pass"]] = r
        first = [passes[min(passes)] for passes in by_take.values()]
        pairs = [(r, e) for r in first if (e := first_own_event(r)) is not None]
        replay_gap = [abs(e0["latency_offset_ms"] - e1["latency_offset_ms"])
                      for passes in by_take.values() if len(passes) == 2
                      if (e0 := first_own_event(passes[0])) and (e1 := first_own_event(passes[1]))]
        algorithmic = [algorithmic_ms(r, e) for r, e in pairs]
        lines += ["", f"#### {condition}", "", *HEADER,
                  _row("from keyword offset", [e["latency_offset_ms"] for _, e in pairs]),
                  _row("  spotter delay + frame quantisation", algorithmic),
                  _row("  Pi compute + publish",
                       [e["latency_offset_ms"] - a for (_, e), a in zip(pairs, algorithmic)]),
                  _row("from keyword onset", [e["latency_onset_ms"] for _, e in pairs])]
        if condition == "loaded":
            # Recovery waits on the server's once-a-second poll, whose phase differs between
            # the two passes of a take, so both passes are kept here: the audio repeats, the
            # server timing does not.
            pre = [e for r in rows if (e := first_own_event(r)) is not None
                   and e.get("in_flight") and e.get("aborted")]
            lines.append(_row("preemption recovery (both passes)",
                              [max(e["abort_return_ms"], e["server_idle_ms"]) for e in pre
                               if e.get("server_idle_ms") is not None]))
        lines += ["", f"Replay difference: max {max(replay_gap):.1f} ms, mean "
                  f"{sum(replay_gap) / len(replay_gap):.1f} ms over {len(replay_gap)} takes.", ""]
        for cls in ("swarm_hold", "swarm_abort"):
            takes = [r for r in first if r["class"] == cls]
            detected = sum(first_own_event(r) is not None for r in takes)
            lines.append(f"- {cls}: detected {_ci(detected, len(takes))}")
        confused = sum(any(e["intent"] == OTHER[r["class"]] for e in r["events"]) for r in first)
        lines.append(f"- other class fired: {_ci(confused, len(first))}")
        if condition == "loaded":
            in_flight = [e for _, e in pairs if e.get("in_flight")]
            cancelled = sum(bool(e.get("aborted")) for e in in_flight)
            lines.append(f"- decode in flight at the trigger and cancelled: "
                         f"{_ci(cancelled, len(in_flight))}")
    return lines


def run() -> int:
    text = "\n".join(exp1_section() + parse_section() + reflex_section()) + "\n"
    OUT_MD.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
