#!/usr/bin/env python3
"""Single-path baseline: the 40 recorded stop takes through the parse path alone (B-M3a).

The ingenieur article's premise is that the parse path is too slow to stop a swarm, so
stop commands need a reflex path of their own. The only evidence so far compares
different audio under different anchors: the parse path on the long golden commands
(Exp-2 `branch_b`) against the reflex path on the short keyword takes (Exp-2 `idle`).
This script removes that confound. Every one of the author's 40 takes (`swarm hold`,
`swarm abort`) goes through the deployed parse path --- Silero endpointing, whisper,
the grammar-constrained parser, validation, the bus and the FSM --- with the reflex
path removed, and latency is measured from the same keyword offset Exp-2's reflex
rows use (`session_meta.json`, the capture tool's speech bounds).

**The harness is Exp-2's.** Same stream loop, core split (front core 0, back cores
1-3), bus, subscriber and airborne FSM per trial; the only change is a Branch A that
never fires, so no reflex command pre-empts the decode or reaches the FSM first. The
spotter's own CPU cost is therefore absent, which favours the single path slightly.

**What a correct stop is.** `swarm hold` must dispatch `{"intent":"hover"}` and
`swarm abort` `{"intent":"abort"}`, scored by Exp-3's rule (every dispatched command
correct). Because either one stops the swarm, a take is also counted as *stopped* if
its first applied command was `hover` or `abort`, whichever was spoken.

**Host.** Latency is of record on the Pi only (`drone-pi`, governor `performance`).
On any other host the stage timings are printed but labelled; the endpointing delay,
the transcripts and the dispatched commands do not depend on the host.

Usage:
    python eval/single_path.py                  # run (on the Pi), then analyse
    python eval/single_path.py --analyse-only
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):  # `python eval/single_path.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from eval.stats import nearest_rank, order_statistic_ci

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT_MD = RESULTS / "single_path.md"
PI_HOST = "drone-pi"

GOLD = {"swarm_hold": '{"intent":"hover"}', "swarm_abort": '{"intent":"abort"}'}
STOP_INTENTS = {"swarm_hold": "hover", "swarm_abort": "abort"}


def raw_path(host: str) -> Path:
    """One file per host, so a workstation run never overwrites the Pi's."""
    return RESULTS / ("single_path_raw.jsonl" if host == PI_HOST else f"single_path_raw_{host}.jsonl")


class NoReflex:
    """A Branch A that never fires: the parse path is the only path."""

    def poll(self, frame: object) -> None:
        return None

    def reset(self) -> None:
        pass


def run_take(trial: dict, runtime, endpointer, subscriber) -> dict:
    from eval.exp2 import BACK, FRONT, PAD_S, soc_state
    from eval.exp3 import TAIL_S, score_segment
    from runtime.stream import StreamLoop, WavSource

    endpointer.reset()
    subscriber.fly()
    # Exp-2's lead pad, so kw_offset_s is the same anchor; Exp-3's tail closes the segment.
    pad = np.zeros(int(PAD_S * 16_000), dtype=np.int16)
    tail = np.zeros(int(TAIL_S * 16_000), dtype=np.int16)
    source = WavSource(np.concatenate([pad, trial["audio"], tail]))
    soc = soc_state()
    result = StreamLoop(runtime, endpointer, front_cores=FRONT, back_cores=BACK).run(source)
    pad_s = endpointer._pad / 16_000
    t_offset = source.t_start + trial["kw_offset_s"]
    segments = []
    for record in result.branch_b:
        u, trace = record.utterance, record.trace
        seg = {"start_s": round(u.start_s, 4), "end_s": round(u.end_s, 4), "t0_s": round(u.t0_s, 4),
               "forced": u.forced, "transcript": trace.transcript, "raw": trace.raw,
               # endpointing declared this long after the keyword's offset (host-independent)
               "endpoint_after_offset_ms": (u.t0_s - trial["kw_offset_s"]) * 1000,
               "vad_wait_ms": 0.0 if u.forced else (u.t0_s - (u.end_s - pad_s)) * 1000,
               "stt_ms": trace.stt_ms, "prefill_ms": trace.prefill_ms, "decode_ms": trace.decode_ms}
        if trace.raw is not None:
            seg.update(score_segment(trace.raw, GOLD[trial["class"]]))
        else:
            seg.update({"canonical": None, "dispatched": None, "correct": False})
        applied = None if trace.message is None else subscriber.wait_for(trace.message.seq)
        if applied is not None:
            seg["applied"] = applied["applied"]
            seg["latency_offset_ms"] = (applied["t_applied"] - t_offset) * 1000
            seg["latency_t0_ms"] = (applied["t_applied"] - record.t0_wall) * 1000
        segments.append(seg)
    return {"take": trial["take"], "class": trial["class"], "kw_offset_s": trial["kw_offset_s"],
            "kw_onset_s": trial["kw_onset_s"], "segments": segments, "soc": soc}


def run(limit: int | None) -> Path:
    from eval.exp2 import BACK, BUS_PORT, FRONT, GAP_S, PORT, Subscriber, _read, keyword_trials
    from eval.exp3 import GGUF
    from runtime.bus import CommandBus
    from runtime.parser import CommandParser
    from runtime.pipeline import PipelineRuntime
    from runtime.stream import pin
    from runtime.vad import Endpointer

    host = platform.node()
    governor = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    pin(BACK)  # llama-server inherits this mask (eval/exp2.py)
    parser, server = CommandParser.for_gguf(GGUF, port=PORT)
    pin(FRONT)
    bus = CommandBus(port=BUS_PORT, bind=True)
    subscriber = Subscriber(bus)
    rows = []
    try:
        runtime = PipelineRuntime(bus=bus, branch_a=NoReflex(), parser=parser)
        endpointer = Endpointer()
        for trial in keyword_trials(1, limit):
            row = run_take(trial, runtime, endpointer, subscriber)
            row["host"], row["governor"] = host, governor
            rows.append(row)
            time.sleep(GAP_S)
    finally:
        subscriber.close()
        bus.close()
        server.stop()
    path = raw_path(host)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(REPO)} ({len(rows)} takes on {host}, governor {governor})")
    return path


# --- analysis ------------------------------------------------------------------


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _pcts(values: Sequence[float]) -> str:
    if not values:
        return "-- | --"
    out = []
    for q in (50, 95):
        lo, hi = order_statistic_ci(values, q)
        fmt = lambda v: "--" if not np.isfinite(v) else f"{v:,.0f}"
        out.append(f"{nearest_rank(values, q):,.0f} [{fmt(lo)}, {fmt(hi)}]")
    return " | ".join(out)


def first_stop(row: dict) -> dict | None:
    """The first segment whose command reached the FSM as the spoken stop."""
    want = STOP_INTENTS[row["class"]]
    return next((s for s in row["segments"]
                 if s.get("dispatched") == want and s.get("applied") and "latency_offset_ms" in s), None)


def reflex_pass0() -> list[float]:
    """Exp-2 idle, first pass: the reflex path's latency from the same anchor."""
    from eval.exp2 import first_own_event

    path = RESULTS / "exp2_raw" / "idle.jsonl"
    out = []
    for row in _read_rows(path):
        event = first_own_event(row) if row["pass"] == 0 else None
        if event is not None:
            out.append(event["latency_offset_ms"])
    return out


def pi_golden_stops() -> list[float]:
    """Exp-2 `branch_b` on the Pi: T0-to-applied for golden items whose gold is a bare stop."""
    path = RESULTS / "exp2_preds" / "clean.jsonl"
    stops = {'{"intent":"hover"}', '{"intent":"abort"}'}
    return [s["e2e_t0_ms"] for r in _read_rows(path) if r["gold"] in stops
            for s in r["segments"] if "e2e_t0_ms" in s]


def analyse() -> int:
    from eval.exp3 import outcome
    from eval.wake_real import clopper_pearson

    candidates = sorted(RESULTS.glob("single_path_raw*.jsonl"))
    if not candidates:
        raise SystemExit("no single_path_raw*.jsonl; run the experiment first")
    path = RESULTS / "single_path_raw.jsonl" if (RESULTS / "single_path_raw.jsonl").is_file() \
        else candidates[0]
    rows = _read_rows(path)
    host = rows[0]["host"]
    of_record = host == PI_HOST
    n = len(rows)

    correct = sum(outcome(r["segments"]).correct for r in rows)
    stopped = [first_stop(r) for r in rows]
    n_stopped = sum(s is not None for s in stopped)
    wrong = [(r["take"], [s["transcript"] for s in r["segments"]], [s["dispatched"] for s in r["segments"]])
             for r in rows if not outcome(r["segments"]).correct]
    c_lo, c_hi = clopper_pearson(n - correct, n)
    s_lo, s_hi = clopper_pearson(n - n_stopped, n)

    lat = [s["latency_offset_ms"] for s in stopped if s is not None]
    endpoint = [r["segments"][0]["endpoint_after_offset_ms"] for r in rows if r["segments"]]
    stt = [s["stt_ms"] for r in rows for s in r["segments"] if s.get("stt_ms") is not None]
    llm = [s["prefill_ms"] + s["decode_ms"] for r in rows for s in r["segments"]
           if s.get("prefill_ms") is not None]
    reflex = reflex_pass0()
    golden_stops = pi_golden_stops()

    label = "" if of_record else f" -- host `{host}`, **not of record**"
    lines = [
        "### Single-path baseline: the stop takes through the parse path (B-M3a)",
        "",
        "Generated by `python eval/single_path.py`. Do not edit by hand.",
        "",
        f"{n} recorded takes ({sum(r['class'] == 'swarm_hold' for r in rows)} `swarm hold`, "
        f"{sum(r['class'] == 'swarm_abort' for r in rows)} `swarm abort`), one pass, reflex path removed; "
        f"source `{path.relative_to(REPO)}`, host `{host}`, governor `{rows[0].get('governor') or '?'}`. "
        "Latency is from keyword offset (the capture tool's speech bounds, Exp-2's reflex anchor) "
        "to the FSM applying the command. Intervals: exact for proportions, order statistics "
        "for percentiles.",
        "",
        "#### Recognition",
        "",
        f"- Correct by Exp-3's rule: **{correct}/{n}**; miss rate {(n - correct) / n:.3f} "
        f"[{c_lo:.3f}, {c_hi:.3f}].",
        f"- Stopped (first applied command was the spoken stop): **{n_stopped}/{n}**; miss rate "
        f"{(n - n_stopped) / n:.3f} [{s_lo:.3f}, {s_hi:.3f}].",
    ]
    for take, transcripts, dispatched in wrong:
        lines.append(f"- Not correct: {take}: transcripts {transcripts}, dispatched {dispatched}.")
    lines += [
        "",
        "#### Latency (ms)",
        "",
        "| Measure | n | p50 [95% CI] | p95 [95% CI] |",
        "| :--- | ---: | :--- | :--- |",
        f"| Endpointing declared, after keyword offset (host-independent) | {len(endpoint)} "
        f"| {_pcts(endpoint)} |",
        f"| Speech recognition{label} | {len(stt)} | {_pcts(stt)} |",
        f"| Prefill + decode{label} | {len(llm)} | {_pcts(llm)} |",
        f"| **Single path, keyword offset to applied**{label} | {len(lat)} | {_pcts(lat)} |",
        f"| Reflex path, keyword offset to published (Exp-2 idle, pass 0, `drone-pi`) | {len(reflex)} "
        f"| {_pcts(reflex)} |",
        f"| Parse path on golden bare-stop items, T0 to applied (Exp-2 `branch_b`, `drone-pi`) "
        f"| {len(golden_stops)} | {_pcts(golden_stops)} |",
        "",
    ]
    if not of_record:
        fastest = (min(endpoint) + min(golden_stops)) if endpoint and golden_stops else float("nan")
        lines += [
            "**Pi figure pending.** The single-path row above was timed on the workstation. "
            "On the Pi the parse path's stages after endpointing were measured on the golden "
            "items whose gold command is a bare `hover` or `abort` (last row); their outputs are "
            "the same commands, their transcripts a few tokens longer. Adding the fastest "
            "endpointing delay here to the fastest of those gives an optimistic Pi estimate of "
            f"**{fastest:,.0f} ms** from keyword offset, against the reflex path's p95 of "
            f"{nearest_rank(reflex, 95):,.0f} ms. It is an estimate, not a measurement: run "
            "`python eval/single_path.py` on the Pi for the figure of record.",
            "",
        ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    print(f"  correct {correct}/{n}, stopped {n_stopped}/{n}; endpoint p50 "
          f"{nearest_rank(endpoint, 50):.0f} ms; single path p50/p95 "
          f"{nearest_rank(lat, 50):.0f}/{nearest_rank(lat, 95):.0f} ms on {host}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N takes (smoke)")
    args = ap.parse_args(argv)
    if not args.analyse_only:
        run(args.limit)
    return analyse()


if __name__ == "__main__":
    sys.exit(main())
