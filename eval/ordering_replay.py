#!/usr/bin/env python3
"""Both paths on the 40 stop takes: what happens to the parse of the stop phrase itself.

Exp-2's keyword trials switch the parse path off, and `eval/single_path.py` switches the
reflex path off, so neither run shows the two together. Here every take goes through the
full dual path, in real time, on Exp-2's harness: same stream loop, core split, bus,
subscriber and airborne FSM per trial, with the trained spotter and the deployed parse path
both live. One thing varies, where the parse path's sequence number is reserved:

  acceptance  when the worker accepts the utterance, after endpointing (the old behaviour,
              reproduced by `ReserveAtAcceptance`, which ignores the number reserved at onset)
  onset       at the endpointer's speech-start event (`runtime/stream.py`)

**What is counted.** For each take the reflex path detected, the fate of each parse the take
produced: `no_command` (nothing published), `discarded` (older than the stop, the ordering
rule dropped it), `rejected` (reached the state machine, which refused it), `repeated`
(applied, same command as the stop), `changed` (applied, a different command: the hold was
replaced). A parse *applied after the stop* is `repeated` or `changed`.

Usage (on the Pi, governor `performance`):
    python eval/ordering_replay.py
    python eval/ordering_replay.py --analyse-only
    python eval/ordering_replay.py --limit 2 --out /tmp/x --any-governor   # smoke
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):  # `python eval/ordering_replay.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from eval.stats import nearest_rank, order_statistic_ci
from runtime.pipeline import PipelineRuntime

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
PI_HOST = "drone-pi"
MODES = ("acceptance", "onset")
RAW_NAME = "ordering_replay_raw.jsonl"
MD_NAME = "ordering_replay.md"

GOLD = {"swarm_hold": '{"intent":"hover"}', "swarm_abort": '{"intent":"abort"}'}
STOP_INTENTS = {"swarm_hold": "hover", "swarm_abort": "abort"}


class ReserveAtAcceptance(PipelineRuntime):
    """The old rule: the number reserved at onset is dropped and a newer one drawn on acceptance."""

    def process(self, audio_path: Path, *, seq: int | None = None):
        return super().process(audio_path)


def fate(seg: dict, stop_intent: str) -> str:
    """What became of one parse, given that a stop of `stop_intent` was published in its take."""
    if seg.get("published_intent") is None:
        return "no_command"
    if seg.get("stale_discarded"):
        return "discarded"
    if not seg.get("applied"):
        return "rejected"
    return "repeated" if seg["published_intent"] == stop_intent else "changed"


def run_take(trial: dict, mode: str, runtime, endpointer, subscriber) -> dict:
    from eval.exp2 import BACK, FRONT, PAD_S, soc_state
    from eval.exp3 import TAIL_S, score_segment
    from runtime.stream import StreamLoop, WavSource
    from train.train_wake import SEED

    np.random.seed(SEED)                 # openwakeword's reset() draws from np.random
    runtime.branch_a.reset()
    endpointer.reset()
    subscriber.fly()
    pad = np.zeros(int(PAD_S * 16_000), dtype=np.int16)
    tail = np.zeros(int(TAIL_S * 16_000), dtype=np.int16)
    source = WavSource(np.concatenate([pad, trial["audio"], tail]))
    soc = soc_state()
    result = StreamLoop(runtime, endpointer, front_cores=FRONT, back_cores=BACK).run(source)
    t_start = source.t_start
    pad_s = endpointer._pad / 16_000
    events = []
    for event in result.branch_a:
        applied = subscriber.wait_for(event.message.seq)
        events.append({"intent": event.message.command["intent"], "seq": event.message.seq,
                       "stream_s": round(event.stream_s, 4),
                       "latency_offset_ms": (event.message.t_publish - (t_start + trial["kw_offset_s"])) * 1000,
                       "latency_onset_ms": (event.message.t_publish - (t_start + trial["kw_onset_s"])) * 1000,
                       "t_applied": None if applied is None else applied["t_applied"]})
    segments = []
    for record in result.branch_b:
        u, trace = record.utterance, record.trace
        seg = {"start_s": round(u.start_s, 4), "speech_start_s": round(u.start_s + pad_s, 4),
               "t0_s": round(u.t0_s, 4), "seq": trace.seq, "transcript": trace.transcript, "raw": trace.raw}
        if trace.raw is not None:
            seg.update(score_segment(trace.raw, GOLD[trial["class"]]))
        if trace.message is not None:
            seg["published_intent"] = trace.message.command["intent"]
            applied = subscriber.wait_for(trace.message.seq)
            if applied is not None:
                seg.update(applied=applied["applied"], stale_discarded=applied["stale_discarded"],
                           latency_offset_ms=(applied["t_applied"] - (t_start + trial["kw_offset_s"])) * 1000)
        segments.append(seg)
    return {"mode": mode, "take": trial["take"], "class": trial["class"], "kw_offset_s": trial["kw_offset_s"],
            "kw_onset_s": trial["kw_onset_s"], "events": events, "segments": segments,
            "final_state": subscriber.dispatcher.fsm.state.value, "soc": soc}


def run(limit: int | None, results: Path, any_governor: bool) -> None:
    from eval.exp2 import BACK, BUS_PORT, FRONT, GAP_S, PORT, Subscriber, _read, keyword_trials
    from eval.exp3 import GGUF
    from runtime.branch_a import load_trained
    from runtime.bus import CommandBus
    from runtime.parser import CommandParser
    from runtime.stream import pin
    from runtime.vad import Endpointer

    governor = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor != "performance" and not any_governor:
        raise SystemExit(f"governor is {governor!r}; Table 7's clock assumption needs 'performance'")
    host = platform.node()
    pin(BACK)  # llama-server inherits this mask (eval/exp2.py)
    parser, server = CommandParser.for_gguf(GGUF, port=PORT)
    pin(FRONT)
    bus = CommandBus(port=BUS_PORT, bind=True)
    subscriber = Subscriber(bus)
    rows = []
    try:
        branch_a = load_trained()
        runtimes = {"acceptance": ReserveAtAcceptance(bus=bus, branch_a=branch_a, parser=parser),
                    "onset": PipelineRuntime(bus=bus, branch_a=branch_a, parser=parser)}
        endpointer = Endpointer()
        for mode in MODES:
            for trial in keyword_trials(1, limit):
                row = run_take(trial, mode, runtimes[mode], endpointer, subscriber)
                row["host"], row["governor"] = host, governor
                rows.append(row)
                print(f"{mode} {row['take']} {row['final_state']} "
                      f"{[s['transcript'] for s in row['segments']]}", flush=True)
                time.sleep(GAP_S)
    finally:
        subscriber.close()
        bus.close()
        server.stop()
    path = results / RAW_NAME
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} trials on {host}, governor {governor})")


# --- analysis ------------------------------------------------------------------


def first_own_event(row: dict) -> dict | None:
    want = STOP_INTENTS[row["class"]]
    return next((e for e in row["events"] if e["intent"] == want), None)


def tally(rows: Sequence[dict]) -> dict:
    """One mode's takes: the fate of each take's parses after its detected stop."""
    detected = [r for r in rows if first_own_event(r) is not None]
    fates: Counter[str] = Counter()
    changed, after_stop_takes = [], []
    before_endpoint = before_onset = 0
    for row in detected:
        event, stop = first_own_event(row), STOP_INTENTS[row["class"]]
        take_fates = [fate(s, stop) for s in row["segments"]]
        fates.update(take_fates)
        if any(f in ("repeated", "changed") for f in take_fates):
            after_stop_takes.append(row["take"])
        for seg, f in zip(row["segments"], take_fates):
            if f == "changed":
                changed.append((row["take"], seg["transcript"], seg["published_intent"]))
        if row["segments"]:
            before_endpoint += event["stream_s"] <= row["segments"][0]["t0_s"]
            before_onset += event["stream_s"] <= row["segments"][0]["speech_start_s"]
    return {"n": len(rows), "detected": len(detected), "fates": fates, "after_stop": len(after_stop_takes),
            "changed": changed, "before_endpoint": before_endpoint, "before_onset": before_onset,
            "missed": [r for r in rows if first_own_event(r) is None]}


def _pcts(values: Sequence[float]) -> str:
    out = []
    for q in (50, 95):
        lo, hi = order_statistic_ci(values, q)
        fmt = lambda v: "--" if not np.isfinite(v) else f"{v:,.0f}"
        out.append(f"{nearest_rank(values, q):,.0f} [{fmt(lo)}, {fmt(hi)}]")
    return " | ".join(out)


def analyse(results: Path) -> int:
    from eval.wake_real import clopper_pearson

    path = results / RAW_NAME
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    host = rows[0]["host"]
    by_mode = {m: [r for r in rows if r["mode"] == m] for m in MODES}
    tallies = {m: tally(by_mode[m]) for m in MODES}
    temps = [r["soc"]["temp"] for r in rows if r["soc"].get("temp") is not None]
    flags = Counter(r["soc"].get("throttled", "?") for r in rows)
    lines = ["### Both paths on the stop takes: the parse of the stop phrase itself (B-M1v, P12)", "",
             "Generated by `python eval/ordering_replay.py`. Do not edit by hand.", "",
             f"{len(by_mode['onset'])} recorded takes per mode, one pass, real time, reflex and parse "
             f"paths both live; host `{host}`, governor `{rows[0].get('governor') or '?'}`"
             + ("" if host == PI_HOST else ", **not of record**") + ". "
             f"Mode `acceptance` reserves the parse path's number after endpointing (the old rule), "
             "`onset` at the endpointer's speech-start event. Intervals: exact (Clopper-Pearson), "
             "take as the unit.", "",
             "| Parse of the stop phrase, on takes the reflex path detected | acceptance | onset |",
             "| :--- | ---: | ---: |",
             f"| Takes detected by the reflex path | {tallies['acceptance']['detected']} "
             f"| {tallies['onset']['detected']} |"]
    for key, label in (("repeated", "applied after the stop, same command"),
                       ("changed", "applied after the stop, a different command"),
                       ("rejected", "published, refused by the state machine"),
                       ("discarded", "published, discarded by the ordering rule"),
                       ("no_command", "nothing published")):
        lines.append(f"| {label} | {tallies['acceptance']['fates'][key]} | {tallies['onset']['fates'][key]} |")
    cells = []
    for m in MODES:
        t = tallies[m]
        lo, hi = clopper_pearson(t["after_stop"], t["detected"])
        cells.append(f"**{t['after_stop']}/{t['detected']}** [{lo:.3f}, {hi:.3f}]")
    lines.append(f"| **Takes with a parse applied after the stop** | {cells[0]} | {cells[1]} |")
    lines.append(f"| Stop published before endpointing closed the utterance | "
                 f"{tallies['acceptance']['before_endpoint']} | {tallies['onset']['before_endpoint']} |")
    lines.append(f"| Stop published before the speech-start event | "
                 f"{tallies['acceptance']['before_onset']} | {tallies['onset']['before_onset']} |")
    lines.append("")
    for m in MODES:
        t = tallies[m]
        for take, transcript, intent in t["changed"]:
            final = next(r["final_state"] for r in by_mode[m] if r["take"] == take)
            lines.append(f"- `{m}`, different command applied: {take}: {transcript!r} -> `{intent}` "
                         f"(final state {final}).")
        for row in t["missed"]:
            outcome = [(s["transcript"], s.get("published_intent"), s.get("applied")) for s in row["segments"]]
            lines.append(f"- `{m}`, reflex path missed: {row['take']}: parse path {outcome}.")
    aborted = {m: Counter(r["final_state"] for r in by_mode[m] if r["class"] == "swarm_abort") for m in MODES}
    lines += ["", f"Final state on abort takes: acceptance {dict(aborted['acceptance'])}, "
              f"onset {dict(aborted['onset'])}.", "", "#### Reflex path with the parse path on (ms, Pi)", "",
              "| Mode | n | keyword offset, p50 [95% CI] | keyword offset, p95 [95% CI] |",
              "| :--- | ---: | :--- | :--- |"]
    for m in MODES:
        values = [e["latency_offset_ms"] for r in by_mode[m] if (e := first_own_event(r)) is not None]
        lines.append(f"| `{m}` | {len(values)} | {_pcts(values)} |")
    lines += ["", f"SoC at trial start: {min(temps):.1f}-{max(temps):.1f} C; throttle word {dict(flags)}." if temps
              else "", ""]
    (results / MD_NAME).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {results / MD_NAME}")
    for m in MODES:
        print(f"  {m}: {tallies[m]['after_stop']}/{tallies[m]['detected']} applied after the stop; "
              f"{dict(tallies[m]['fates'])}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N takes (smoke)")
    ap.add_argument("--out", type=Path, default=RESULTS, help="results directory (smoke runs)")
    ap.add_argument("--any-governor", action="store_true", help="skip the governor check (off the Pi)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.analyse_only:
        run(args.limit, args.out, args.any_governor)
    return analyse(args.out)


if __name__ == "__main__":
    sys.exit(main())
