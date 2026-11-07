#!/usr/bin/env python3
"""Voice-onset hold: a MIRA-style baseline for the reflex path (B-M3b, P12b).

A hold published when the endpointer detects speech onset, with no keyword. It needs no
spotter and passes the article's own membership rule, since a false hold can only make the
swarm do less. So the question it puts is what the keyword buys.

**Latency.** The same 40 takes and anchors as Exp-2's reflex rows, in real time on the
board: the frame loop on core 0 runs the endpointer alone, and the hold goes through the same
bus and subscriber. The parser is not resident and nothing loads cores 1-3, so this is the
idle condition. Measured on the Pi only; the analysis times it from the forced-aligned
keyword (P15, `eval/anchors.py`), the trial logs from the capture tool's speech bounds.

**False holds.** Every speech onset on a stream that is not a stop command, on the two
streams `eval/wake_delay_curve.py` counts the spotter on: the 200 golden recordings, 0.5 s
apart, and the held-out ambient stream. Golden items whose own command is a bare hold or abort
are counted apart, since a hold there is right. The count runs through the endpointer alone,
faster than real time, and does not depend on the host. The ambient clips are not on the
Pi, so that stream is counted on the workstation; the golden stream is counted on both, as a
check of that claim. Counts are given per onset, and with the 1 s debounce the spotter's
counts use. Intervals are Poisson.

Usage:
    python eval/voice_onset.py --latency          # on the Pi, governor `performance`
    python eval/voice_onset.py --count            # either host; streams it has
    python eval/voice_onset.py --analyse-only     # on the workstation
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):  # `python eval/voice_onset.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from eval.stats import nearest_rank, order_statistic_ci

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT_MD = "voice_onset.md"
RAW_NAME = "voice_onset_raw.jsonl"
PI_HOST = "drone-pi"
STOP_GOLD = {'{"intent":"hover"}', '{"intent":"abort"}'}
DEBOUNCE_S = 1.0   # train_wake.py's


def debounced(times: Sequence[float], debounce_s: float = DEBOUNCE_S) -> int:
    """Rising edges closer than `debounce_s` merge, as `train_wake.count_events` counts them."""
    if not len(times):
        return 0
    return 1 + int(np.sum(np.diff(np.asarray(times)) > debounce_s))


def onset_times(audio: np.ndarray, endpointer=None) -> list[float]:
    """Stream time at the end of each window that opened a segment."""
    from runtime.stream import FRAME
    from runtime.vad import RATE, Endpointer

    endpointer = endpointer or Endpointer()
    endpointer.reset()
    times: list[float] = []
    endpointer.on_speech_start = lambda: times.append(endpointer._consumed / RATE)
    for i in range(0, audio.size - FRAME + 1, FRAME):
        endpointer.feed(audio[i : i + FRAME])
    return times


def item_index(starts: Sequence[float], t: float) -> int:
    """The golden item (or the gap after it) that stream time `t` falls in."""
    return int(np.searchsorted(np.asarray(starts), t, side="right") - 1)


# --- latency, on the board -----------------------------------------------------


class NoSpotter:
    """`StreamLoop`'s runtime with no reflex path: this baseline has no keyword."""

    def on_wake_frame(self, frame: object) -> None:
        return None


def run_latency(limit: int | None, results: Path, any_governor: bool) -> None:
    from eval.exp2 import BUS_PORT, FRONT, GAP_S, PAD_S, Subscriber, _read, keyword_trials, soc_state
    from eval.exp3 import TAIL_S
    from runtime.bus import CommandBus
    from runtime.stream import StreamLoop, WavSource, pin
    from runtime.vad import RATE, Endpointer
    from schema.schema import Hover

    governor = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor != "performance" and not any_governor:
        raise SystemExit(f"governor is {governor!r}; Table 7's clock assumption needs 'performance'")
    host = platform.node()
    pin(FRONT)
    bus = CommandBus(port=BUS_PORT, bind=True)
    subscriber = Subscriber(bus)
    endpointer = Endpointer()
    rows = []
    try:
        for trial in keyword_trials(1, limit):
            endpointer.reset()
            subscriber.fly()
            holds: list[dict] = []

            def hold() -> None:
                message = bus.publish("A", Hover(intent="hover"))
                holds.append({"seq": message.seq, "t_publish": message.t_publish,
                              "stream_s": endpointer._consumed / RATE})

            endpointer.on_speech_start = hold
            pad = np.zeros(int(PAD_S * RATE), dtype=np.int16)
            tail = np.zeros(int(TAIL_S * RATE), dtype=np.int16)
            source = WavSource(np.concatenate([pad, trial["audio"], tail]))
            soc = soc_state()
            result = StreamLoop(NoSpotter(), endpointer, branch_b=False, front_cores=FRONT).run(source)
            events = []
            for h in holds:
                applied = subscriber.wait_for(h["seq"])
                events.append({"seq": h["seq"], "stream_s": round(h["stream_s"], 4),
                               "latency_onset_ms": (h["t_publish"] - (source.t_start + trial["kw_onset_s"])) * 1000,
                               "latency_offset_ms": (h["t_publish"] - (source.t_start + trial["kw_offset_s"])) * 1000,
                               "algorithmic_onset_ms": (h["stream_s"] - trial["kw_onset_s"]) * 1000,
                               "fsm_ms": None if applied is None else (applied["t_applied"] - h["t_publish"]) * 1000,
                               "applied": None if applied is None else applied["applied"]})
            rows.append({"take": trial["take"], "class": trial["class"], "kw_onset_s": trial["kw_onset_s"],
                         "kw_offset_s": trial["kw_offset_s"], "events": events, "soc": soc, "host": host,
                         "governor": governor, "frame_ms_max": max(f.compute_s for f in result.frames) * 1000})
            time.sleep(GAP_S)
    finally:
        subscriber.close()
        bus.close()
    path = results / RAW_NAME
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} takes on {host}, governor {governor})")


# --- false holds ---------------------------------------------------------------


def run_count(results: Path) -> None:
    from runtime.vad import RATE, Endpointer
    from train.train_wake import (GOLDEN, GOLDEN_AUDIO, MANIFEST, _load_int16, golden_stream,
                                  negative_stream, poisson_ci)

    host = platform.node()
    endpointer = Endpointer()
    record: dict = {"host": host}

    rows = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    audio = golden_stream()
    starts, at = [], 0.0
    for row in rows:
        starts.append(at)
        at += _load_int16(GOLDEN_AUDIO / f"{row['id']}.wav").size / RATE + 0.5
    times = onset_times(audio, endpointer)
    hours = audio.size / RATE / 3600
    items = [item_index(starts, t) for t in times]
    false = [t for t, i in zip(times, items) if rows[i]["target"] not in STOP_GOLD]
    lo, hi = poisson_ci(len(false), hours)
    record["golden"] = {"hours": hours, "items": len(rows), "onsets": len(times),
                        "stop_items": sum(r["target"] in STOP_GOLD for r in rows),
                        "onsets_on_stop_items": len(times) - len(false), "false_holds": len(false),
                        "false_holds_debounced": debounced(false),
                        "items_with_a_hold": len(set(items)),
                        "non_stop_items_with_a_hold": len({i for i in items if rows[i]["target"] not in STOP_GOLD}),
                        "per_hour": len(false) / hours, "ci95": [float(lo), float(hi)]}
    print(f"golden on {host}: {record['golden']}", flush=True)

    clips = json.loads(MANIFEST.read_text())["clips"]
    try:
        audio = negative_stream(clips, "test", True)
    except (OSError, RuntimeError) as exc:     # the Pi has the manifest, not the clips
        print(f"ambient stream absent on {host}: {exc}")
    else:
        times = onset_times(audio, endpointer)
        hours = audio.size / RATE / 3600
        lo, hi = poisson_ci(len(times), hours)
        record["ambient"] = {"hours": hours, "false_holds": len(times), "false_holds_debounced": debounced(times),
                             "per_hour": len(times) / hours, "ci95": [float(lo), float(hi)]}
        print(f"ambient on {host}: {record['ambient']}", flush=True)
    path = results / f"voice_onset_counts_{host}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


# --- report --------------------------------------------------------------------


def _pcts(values: Sequence[float]) -> str:
    out = []
    for q in (50, 95):
        lo, hi = order_statistic_ci(values, q)
        fmt = lambda v: "--" if not np.isfinite(v) else f"{v:,.0f}"
        out.append(f"{nearest_rank(values, q):,.0f} [{fmt(lo)}, {fmt(hi)}]")
    return " | ".join(out)


def _reflex_pass0(results: Path) -> list[dict]:
    from eval.exp2 import aligned_latencies

    rows = [json.loads(l) for l in (results / "exp2_raw" / "idle.jsonl").read_text().splitlines() if l.strip()]
    return aligned_latencies(rows)


def analyse(results: Path) -> int:
    from eval.wake_real import clopper_pearson

    rows = [json.loads(l) for l in (results / RAW_NAME).read_text(encoding="utf-8").splitlines() if l.strip()]
    host = rows[0]["host"]
    from eval.anchors import take_shift_ms

    first = [{kind: r["events"][0][f"latency_{kind}_ms"] + take_shift_ms(r["take"], kind)
              for kind in ("onset", "offset")} for r in rows if r["events"]]
    detected = len(first)
    d_lo, d_hi = clopper_pearson(len(rows) - detected, len(rows))
    extra = sum(len(r["events"]) > 1 for r in rows)
    early = sum(e["onset"] < 0 for e in first)
    reflex = _reflex_pass0(results)
    temps = [r["soc"]["temp"] for r in rows if r["soc"].get("temp") is not None]
    flags = sorted({r["soc"].get("throttled", "?") for r in rows})
    counts = {p.stem.removeprefix("voice_onset_counts_"): json.loads(p.read_text())
              for p in sorted(results.glob("voice_onset_counts_*.json"))}
    workstation = next(c for c in counts.values() if "ambient" in c)
    pi_counts = counts.get(PI_HOST)
    curve = json.loads((results / "wake_delay_curve.json").read_text())
    spotter = next(r for r in curve["false_accepts"] if r["threshold"] == curve["operating_point"])
    golden, ambient = workstation["golden"], workstation["ambient"]

    lines = [
        "### Voice-onset hold: what the keyword buys (B-M3b, P12b)", "",
        "Generated by `python eval/voice_onset.py`. Do not edit by hand.", "",
        f"A hold published at the endpointer's speech-start event, no keyword. {len(rows)} recorded takes, one "
        f"pass, same anchors as Exp-2 (forced-aligned keyword, P15); host `{host}`, governor `{rows[0].get('governor') or '?'}`, idle. "
        "Intervals: exact for proportions, order statistics for percentiles, Poisson for false holds.", "",
        "#### Latency to publish (ms)", "",
        "| Path, anchor | n | p50 [95% CI] | p95 [95% CI] |", "| :--- | ---: | :--- | :--- |",
        f"| Voice-onset hold, keyword onset | {detected} | {_pcts([e['onset'] for e in first])} |",
        f"| Voice-onset hold, keyword offset | {detected} | {_pcts([e['offset'] for e in first])} |",
        f"| Reflex path, keyword onset (Exp-2 idle, pass 0) | {len(reflex)} "
        f"| {_pcts([e['onset'] for e in reflex])} |",
        f"| Reflex path, keyword offset (Exp-2 idle, pass 0) | {len(reflex)} "
        f"| {_pcts([e['offset'] for e in reflex])} |", "",
        f"- Takes that produced a hold: **{detected}/{len(rows)}**; miss rate "
        f"{(len(rows) - detected) / len(rows):.3f} [{d_lo:.3f}, {d_hi:.3f}]. Takes with more than one hold: {extra}.",
        f"- Holds before the aligned keyword onset: {early}/{detected}.",
        f"- All {sum(r['class'] == 'swarm_abort' for r in rows)} abort takes get a hold, not an abort: this path "
        "has one output.", "",
        "#### False holds, against the spotter at its operating point", "",
        "| Stream | hours | voice-onset, onsets | per hour [95% CI] | debounced 1 s | spotter events "
        "| spotter per hour [95% CI] |",
        "| :--- | ---: | ---: | :--- | ---: | ---: | :--- |",
        f"| Golden recordings, stop items excluded | {golden['hours']:.3f} | {golden['false_holds']} "
        f"| {golden['per_hour']:,.0f} [{golden['ci95'][0]:,.0f}, {golden['ci95'][1]:,.0f}] "
        f"| {golden['false_holds_debounced']} | {spotter['golden']['events']} "
        f"| {spotter['golden']['per_hour']:.2f} [{spotter['golden']['ci95'][0]:.2f}, "
        f"{spotter['golden']['ci95'][1]:.2f}] |",
        f"| Ambient (held out) | {ambient['hours']:.3f} | {ambient['false_holds']} "
        f"| {ambient['per_hour']:,.0f} [{ambient['ci95'][0]:,.0f}, {ambient['ci95'][1]:,.0f}] "
        f"| {ambient['false_holds_debounced']} | {spotter['ambient']['events']} "
        f"| {spotter['ambient']['per_hour']:.2f} [{spotter['ambient']['ci95'][0]:.2f}, "
        f"{spotter['ambient']['ci95'][1]:.2f}] |", "",
        f"- Golden: {golden['non_stop_items_with_a_hold']} of the "
        f"{golden['items'] - golden['stop_items']} non-stop commands drew at least one hold; "
        f"{golden['onsets_on_stop_items']} further onsets fell on the {golden['stop_items']} stop items and are "
        "not counted as false.",
        f"- Counted on `{workstation['host']}` (the ambient clips are not on the Pi; the count does not depend "
        "on the host)." + (f" Golden stream on `{PI_HOST}`: {pi_counts['golden']['onsets']} onsets, "
                           f"{pi_counts['golden']['false_holds']} false, against {golden['onsets']} and "
                           f"{golden['false_holds']} here." if pi_counts else ""), "",
        f"SoC at trial start: {min(temps):.1f}-{max(temps):.1f} C; throttle word {flags}." if temps else "", "",
    ]
    (results / OUT_MD).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {results / OUT_MD}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--latency", action="store_true", help="run the 40 takes (Pi)")
    ap.add_argument("--count", action="store_true", help="count onsets on the streams this host has")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N takes (smoke)")
    ap.add_argument("--out", type=Path, default=RESULTS, help="results directory (smoke runs)")
    ap.add_argument("--any-governor", action="store_true", help="skip the governor check (off the Pi)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.latency:
        run_latency(args.limit, args.out, args.any_governor)
    if args.count:
        run_count(args.out)
    if args.analyse_only:
        return analyse(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
