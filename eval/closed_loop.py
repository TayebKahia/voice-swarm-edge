#!/usr/bin/env python3
"""Closed loop: a spoken stop, from the operator's mouth to the vehicles' arrest (B-M4, P11).

The chain of Figure 1, run whole: the USB microphone on the Raspberry Pi 5, `runtime/main.py`
(both paths, `--reflex-gate off`, so the consumer's ordering rule is the only protection, as the
article describes it), the bus over Wi-Fi, and on the workstation `demo/workstation.py`: the
ordering rule, the flight state machine, `swarm/link.py`, the controller, and the PyFlyt
rigid-body backend with no separation clamp. The operator speaks a formation change; once the
fastest vehicle passes 1 m/s the workstation cues, and the operator says "swarm hold" or "swarm
abort". Each such stop is a take. Conditions: the Pi online, and the Pi with its default route
and DNS removed, so that only the link to the workstation remains (offline operation).

**Clocks.** Onset and publication are on the Pi's `time.monotonic()`, receipt and the vehicles on
the workstation's. `demo/clock.py` probes one from the other over the same link throughout the run;
the fitted offset carries the bound printed with it, and only the publish-to-apply leg depends on it.

**Utterance onset.** Found in the audio the frame loop consumed (`runtime/main.py --capture`): 10 ms
levels, a floor at the 10th percentile of the six seconds before the trigger, and the onset where the
level last rose 12 dB above that floor before the trigger, with gaps under 0.4 s bridged so that
"swarm, hold" is one region. Sample times come from a line fitted under the frames' delivery stamps,
so they carry the capture chain's own latency, a few milliseconds, which shortens every latency here
by the same amount. The region's length is reported, as a check on the rule. The onset of record is
the forced-aligned start of "swarm" (P15, `eval/anchors.py`), placed on the same sample clock; the
energy region still decides "onset ambiguous", so no take changes outcome class.

**Vehicle quantities**, from every tick of the backend (24 Hz):
    v0         fastest vehicle's speed on the last tick before the stop was applied
    rest       hold: first tick from which every speed stays under 0.1 m/s for 0.5 s;
               abort: the tick on which the motors stopped on ground contact
    distance   largest horizontal displacement of any vehicle from where it was when the stop was
               applied (and, separately, from where it was at utterance onset), until rest
    a_eff      v0^2 / (2 distance): the constant deceleration that would give the measured distance,
               to set against Eq. (eq:stopping)'s a = 4 m/s^2

A take counts only if v0 >= 0.5 m/s, i.e. the stop arrived during the formation change; the others
are counted and excluded. A take whose stop the reflex path missed is counted as a miss, and its
latency is not included.

Usage:
    python eval/closed_loop.py                     # analyse every run under results/closed_loop_raw/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):  # `python eval/closed_loop.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from demo.clock import LocalClock
from demo.clock import fit as fit_clock
from demo.clock import load as load_probes
from eval.stats import nearest_rank, order_statistic_ci

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "closed_loop_raw"
OUT_MD = REPO / "results" / "closed_loop.md"
OUT_TAKES = REPO / "results" / "closed_loop_takes.jsonl"

RATE = 16_000
FRAME = 1_280
LEVEL_WIN = 160                 # 10 ms
FLOOR_S = 6.0
RISE_DB = 12.0
GAP_S = 0.4
SEARCH_S = 4.0
REST_SPEED = 0.1
REST_HOLD_S = 0.5
MOVING_SPEED = 0.5
A_ARTICLE = 4.0
REST_LIMIT_S = 30.0
#: A stop phrase is about a second long; a longer region has merged it with other speech.
MAX_REGION_S = 2.0
#: A parse-path copy of a phrase the reflex caught is published about 2 s after the reflex.
REFLEX_BEFORE_S = 4.0
STOP_WORD = {"hover": "hold", "abort": "abort"}
CONDITIONS = ("online", "offline")
INTENTS = ("hover", "abort")


# ----------------------------------------------------------------------------- audio


def sample_clock(tdue: np.ndarray, frame: int = FRAME, rate: int = RATE) -> tuple[float, float]:
    """(t0, slope): the Pi-clock time of sample i is about t0 + slope * i.

    Stamps are delivery times, never earlier than the frame's last sample, so the line is fitted
    through them and lowered onto their lower envelope.
    """
    end_sample = (np.arange(tdue.size) + 1) * frame - 1
    slope, intercept = np.polyfit(end_sample, tdue, 1)
    intercept += float(np.min(tdue - (slope * end_sample + intercept)))
    return float(intercept), float(slope)


def levels_db(audio: np.ndarray, win: int = LEVEL_WIN) -> np.ndarray:
    n = audio.size // win
    x = audio[: n * win].astype(np.float64).reshape(n, win) / 32768.0
    return 10 * np.log10(np.mean(x * x, axis=1) + 1e-12)


def find_onset(audio: np.ndarray, trigger_sample: int, *, rate: int = RATE) -> tuple[int, float] | None:
    """(onset sample, region seconds) of the speech region that holds `trigger_sample`."""
    lo = max(0, trigger_sample - int(FLOOR_S * rate))
    lev = levels_db(audio[lo:trigger_sample])
    if lev.size < 10:
        return None
    floor = float(np.percentile(lev, 10))
    active = lev > floor + RISE_DB
    search = int(SEARCH_S * rate / LEVEL_WIN)
    gap = int(GAP_S * rate / LEVEL_WIN)
    i = lev.size - 1
    while i >= 0 and not active[i]:            # the trigger fires after speech ends, sometimes
        i -= 1
        if lev.size - 1 - i > search:
            return None
    onset, silent = None, 0
    while i >= 0 and lev.size - 1 - i <= search:
        if active[i]:
            onset, silent = i, 0
        else:
            silent += 1
            if silent > gap:
                break
        i -= 1
    if onset is None:
        return None
    onset_sample = lo + onset * LEVEL_WIN
    return onset_sample, (trigger_sample - onset_sample) / rate


# ----------------------------------------------------------------------------- vehicles


def rest_index(speeds: np.ndarray, start: int, tick_s: float) -> int | None:
    """First index >= start from which every row's max speed stays < REST_SPEED for REST_HOLD_S."""
    need = max(1, int(math.ceil(REST_HOLD_S / tick_s)))
    still = speeds.max(axis=1) < REST_SPEED
    run = 0
    for k in range(start, still.size):
        run = run + 1 if still[k] else 0
        if run >= need:
            return k - need + 1
    return None


def horizontal_excursion(positions: np.ndarray, start: int, stop: int, ref: np.ndarray) -> float:
    d = np.linalg.norm(positions[start:stop + 1, :, :2] - ref[None, :, :2], axis=2)
    return float(d.max()) if d.size else 0.0


@dataclass
class Take:
    run: str
    condition: str
    intent: str
    seq: int
    outcome: str                     # "measured", "not moving", "no onset", "no rest"
    v0: float = float("nan")
    region_s: float = float("nan")
    onset_to_publish_ms: float = float("nan")
    publish_to_apply_ms: float = float("nan")
    apply_to_rest_ms: float = float("nan")
    onset_to_rest_ms: float = float("nan")
    onset_to_apply_ms: float = float("nan")
    distance_from_apply_m: float = float("nan")
    distance_from_onset_m: float = float("nan")
    a_eff: float = float("nan")
    predicted_m: float = float("nan")
    overridden: bool = False
    tick_gap_max_ms: float = float("nan")
    path: str = "reflex"
    clock_bound_ms: float = float("nan")


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_audio(run: Path) -> tuple[np.ndarray, np.ndarray]:
    if (run / "cap.flac").is_file():
        import soundfile as sf

        audio, _ = sf.read(run / "cap.flac", dtype="int16")
    else:
        audio = np.fromfile(run / "cap.pcm", dtype="<i2")
    tdue = np.fromfile(run / "cap.tdue", dtype="<f8")
    return audio[: tdue.size * FRAME], tdue


def analyse_run(run: Path, onsets: dict[tuple[str, str, int], int | None] | None = None
                ) -> tuple[dict, list[Take], dict]:
    """One run's takes. `onsets` maps (run, path, seq) to the aligned onset sample
    (`anchors.closed_loop_onsets()`); without it the energy onset is used."""
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    condition = meta["condition"]
    pi = jsonl(run / "pi.jsonl")
    ws = jsonl(run / "workstation.jsonl")
    probes = load_probes(run / "clock.jsonl")
    clock = fit_clock(probes)
    local = LocalClock(probes)
    audio, tdue = load_audio(run)
    t0, slope = sample_clock(tdue)
    ticks = np.load(run / "ticks.npz" if (run / "ticks.npz").is_file() else run / "ticks.partial.npz")
    mono, pos, vel = ticks["mono"], ticks["positions"], ticks["velocities"]
    motors, tick_s = ticks["motors"], float(ticks["tick_seconds"])
    speeds = np.linalg.norm(vel, axis=2)

    reflexes = [e for e in pi if e.get("event") == "reflex_published"]
    received = [e for e in ws if e.get("event") == "bus_received"]
    by_seq = {(e["branch"], e["seq"]): e for e in received}
    cues = [e for e in ws if e.get("event") == "cue"]
    parse_stops = [e for e in received if e["branch"] == "B" and e["command"].get("intent") in INTENTS]
    reflex_pub = [r["t_publish"] for r in reflexes]
    bounds: list[float] = []

    def measure(path: str, intent: str, seq: int, t_publish: float, trigger_sample: int) -> Take:
        take = Take(run.name, condition, intent, seq, "measured", path=path)
        rec = by_seq.get(("A" if path == "reflex" else "B", seq))
        if rec is None:
            take.outcome = "lost on the link"
            return take
        t_apply = rec["t_handle"]
        k = int(np.searchsorted(mono, t_apply))          # first tick after the stop was applied
        if k == 0 or k >= mono.size:
            take.outcome = "outside the trace"
            return take
        take.v0 = float(speeds[k - 1].max())
        offset, bound = local.at(t_publish)
        bounds.append(bound)
        take.clock_bound_ms = bound * 1000
        take.publish_to_apply_ms = (t_apply - (t_publish + offset)) * 1000
        found = find_onset(audio, trigger_sample)
        if onsets is not None and (run.name, path, seq) in onsets:
            aligned = onsets[(run.name, path, seq)]
            found = None if found is None or aligned is None else (aligned, found[1])
        t_onset_ws = None
        if found is not None:
            onset_sample, take.region_s = found
            t_onset_pi = t0 + slope * onset_sample
            take.onset_to_publish_ms = (t_publish - t_onset_pi) * 1000
            t_onset_ws = local.to_workstation(t_onset_pi)
            take.onset_to_apply_ms = (t_apply - t_onset_ws) * 1000
        if intent == "hover":
            rest = rest_index(speeds, k, tick_s)
        else:
            off = np.flatnonzero(~motors[k:].astype(bool))
            rest = None if off.size == 0 else int(k + off[0])
        if rest is not None and (mono[rest] - t_apply) > REST_LIMIT_S:
            rest = None
        later = [e for e in received
                 if e["t_handle"] > t_apply
                 and e["t_handle"] <= (mono[rest] if rest is not None else t_apply + REST_LIMIT_S)
                 and e["dispatched"] not in (None, intent)]
        take.overridden = bool(later)
        if take.v0 < MOVING_SPEED:
            take.outcome = "not moving"
        elif found is None:
            take.outcome = "no onset"
        elif take.region_s > MAX_REGION_S:
            take.outcome = "onset ambiguous"
        elif rest is None:
            take.outcome = "no rest"
        if rest is not None:
            take.apply_to_rest_ms = (mono[rest] - t_apply) * 1000
            take.distance_from_apply_m = horizontal_excursion(pos, k, rest, pos[k - 1])
            take.tick_gap_max_ms = float(np.diff(mono[k - 1:rest + 1]).max() * 1000) if rest > k else 0.0
            if take.distance_from_apply_m > 0:
                take.a_eff = take.v0 ** 2 / (2 * take.distance_from_apply_m)
            take.predicted_m = take.v0 ** 2 / (2 * A_ARTICLE)
            if t_onset_ws is not None:
                j = int(np.clip(np.searchsorted(mono, t_onset_ws), 1, mono.size - 1))
                take.distance_from_onset_m = horizontal_excursion(pos, j, rest, pos[j - 1])
                take.onset_to_rest_ms = (mono[rest] - t_onset_ws) * 1000
        return take

    takes: list[Take] = []
    for r in reflexes:
        takes.append(measure("reflex", r["intent"], r["seq"], r["t_publish"], (r["frame"] + 1) * FRAME - 1))
    # Stops the parse path delivered after a reflex miss: a stop it published and the consumer applied
    # (not discarded as stale), whose transcript holds the stop word, with no reflex published in the
    # REFLEX_BEFORE_S before it. A hold the validator substituted for a garbled command is not one.
    heard = [e for e in pi if e.get("event") == "parse_heard" and e.get("command")
             and e["command"].get("intent") in INTENTS and e.get("t_publish") is not None]
    for e in heard:
        if any(0 <= e["t_publish"] - t <= REFLEX_BEFORE_S for t in reflex_pub):
            continue
        if not re.search(rf"\b{STOP_WORD[e['command']['intent']]}\b", (e.get("transcript") or "").lower()):
            continue
        rec = by_seq.get(("B", e["seq"]))
        if rec is None or rec["stale"] or rec["dispatched"] != e["command"]["intent"]:
            continue
        trigger = int(round((e["t_start"] - t0) / slope))
        takes.append(measure("parse", e["command"]["intent"], e["seq"], e["t_publish"], trigger))

    # Utterance level: stop phrases the parse path transcribed with the stop word in them, and
    # whether the reflex path fired on the same phrase (a reflex published in the 4 s before).
    words = STOP_WORD
    phrases = []
    for e in pi:
        if e.get("event") != "parse_heard" or e.get("t_publish") is None:
            continue
        text = (e.get("transcript") or "").lower()
        for intent, word in words.items():
            if re.search(rf"\b{word}\b", text):
                fired = any(0 <= e["t_publish"] - t <= REFLEX_BEFORE_S for t in reflex_pub)
                phrases.append({"intent": intent, "fired": fired, "transcript": e["transcript"]})

    reflex_t = [by_seq[("A", r["seq"])]["t_handle"] for r in reflexes if ("A", r["seq"]) in by_seq]
    # Per cue: did a reflex stop arrive between the formation that raised the cue and the next one?
    formations = sorted(e["t_handle"] for e in received if e["dispatched"] == "formation")
    cue_hits = []
    for c in cues:
        start = max((t for t in formations if t <= c["mono"]), default=-math.inf)
        end = min((t for t in formations if t > c["mono"]), default=math.inf)
        hit = [t for t in reflex_t if start < t < end]
        stop = [e["command"]["intent"] for e in parse_stops if start < e["t_handle"] < end]
        cue_hits.append({"hit": bool(hit), "first_reflex_s": (min(hit) - c["mono"]) if hit else None,
                         "parse_stop": stop[0] if stop else None})
    thermal = [line for line in (run / "thermal.log").read_text().splitlines()] if (run / "thermal.log").is_file() else []
    temps = [float(l.split("temp=")[1].split("'")[0]) for l in thermal if "temp=" in l]
    throttled = sorted({l.split("throttled=")[1].split()[0] for l in thermal if "throttled=" in l})
    run_end = next((e for e in pi if e.get("event") == "run_end"), {})
    extra = {
        "run": run.name, "condition": condition, "meta": meta, "clock": clock.as_dict(),
        "cues": len(cues), "reflexes": len(reflexes),
        "reflex_by_intent": {i: sum(r["intent"] == i for r in reflexes) for i in INTENTS},
        "phrases": phrases,
        "local_bound_ms_max": max(bounds) * 1000 if bounds else None,
        "cue_hits": cue_hits,
        "stale_discarded": sum(bool(e["stale"]) for e in received),
        "overflows": run_end.get("overflows"),
        "temp_max": max(temps) if temps else None, "throttled": throttled,
        "reanchored": sum(e.get("event") == "pacing_reanchored" for e in ws),
        "tick_ms_p50": float(np.median(np.diff(mono)) * 1000) if mono.size > 1 else None,
        "tick_s": tick_s,
    }
    return meta, takes, extra


# ----------------------------------------------------------------------------- report


def _fmt(x: float, nd: int = 0) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"{x:,.{nd}f}"


def summary(values: Sequence[float], nd: int = 0) -> str:
    v = [x for x in values if math.isfinite(x)]
    if not v:
        return "-- | -- | --"
    lo, hi = order_statistic_ci(v, 50)
    return (f"{_fmt(float(np.median(v)), nd)} [{_fmt(lo, nd)}, {_fmt(hi, nd)}] | "
            f"{_fmt(nearest_rank(v, 90), nd)} | {_fmt(min(v), nd)}--{_fmt(max(v), nd)}")


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    from scipy.stats import beta

    tail = (1 - confidence) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return lo, hi


ROWS = [
    ("Onset to publication (Pi)", "onset_to_publish_ms", 0),
    ("Publication to applied (Wi-Fi + consumer)", "publish_to_apply_ms", 0),
    ("Onset to applied", "onset_to_apply_ms", 0),
    ("Applied to rest (time to rest)", "apply_to_rest_ms", 0),
    ("**Onset to rest (arrest)**", "onset_to_rest_ms", 0),
    ("Speed when applied, v0 (m/s)", "v0", 2),
    ("Distance from applied (m)", "distance_from_apply_m", 2),
    ("Distance from onset (m)", "distance_from_onset_m", 2),
    ("v0^2/2a, a = 4 m/s^2 (m)", "predicted_m", 2),
    ("Effective deceleration v0^2/2d (m/s^2)", "a_eff", 1),
    ("Stopping-phrase region found (s)", "region_s", 2),
]


def report(runs: list[tuple[dict, list[Take], dict]], onset_note: str | None = None) -> str:
    takes = [t for _, ts, _ in runs for t in ts]
    lines = [
        "### Closed loop: spoken stop to vehicle arrest (B-M4, P11)", "",
        "Generated by `python eval/closed_loop.py`. Do not edit by hand.", "",
        "Live microphone on the Raspberry Pi 5 (`runtime/main.py --reflex-gate off`), bus over Wi-Fi, "
        "consumer, flight state machine, `swarm/link.py` and the PyFlyt rigid-body backend (no "
        "separation clamp) on the workstation. A take is one spoken stop during a formation change. "
        "Latencies in ms. Intervals: order statistics for medians (95%), exact (Clopper-Pearson) for "
        "proportions. Distances are horizontal, the largest over the five vehicles."
        + (f" {onset_note}" if onset_note else ""), "",
        "#### Runs", "",
        "| Run | Condition | Governor | Cues | Reflex triggers (hold/abort) | Stale parses discarded | "
        "Clock bound, run fit / largest per stop (ms) | Tick p50 (ms) | Re-anchors | Max temp (C) | Throttle flags | Overruns |",
        "| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: |",
    ]
    for meta, _, x in runs:
        rb = x["reflex_by_intent"]
        lines.append(
            f"| {x['run']} | {x['condition']} | {meta.get('governor', '--')} | {x['cues']} | "
            f"{x['reflexes']} ({rb['hover']}/{rb['abort']}) | {x['stale_discarded']} | "
            f"{x['clock']['bound_ms']:.1f} / {_fmt(x['local_bound_ms_max'], 1)} | {_fmt(x['tick_ms_p50'], 1)} | {x['reanchored']} | "
            f"{_fmt(x['temp_max'], 1)} | {', '.join(x['throttled']) or '--'} | {_fmt(x['overflows'])} |")
    for condition, intent, path in [(c, i, p) for c in CONDITIONS for i in INTENTS for p in ("reflex", "parse")]:
            cell = [t for t in takes if t.condition == condition and t.intent == intent and t.path == path]
            if not cell:
                continue
            ok = [t for t in cell if t.outcome == "measured"]
            excluded: dict[str, int] = {}
            for t in cell:
                if t.outcome != "measured":
                    excluded[t.outcome] = excluded.get(t.outcome, 0) + 1
            over = sum(t.overridden for t in cell)
            lo, hi = clopper_pearson(over, len(cell))
            name = "Hold" if intent == "hover" else "Abort"
            via = "reflex path" if path == "reflex" else "parse path, after a reflex miss"
            lines += ["", f"#### {name} via the {via}, {condition} (n = {len(ok)} measured of {len(cell)} stops)", "",
                      "| Quantity | median [95% CI] | p90 | range |", "| :--- | :--- | ---: | :--- |"]
            for label, key, nd in ROWS:
                lines.append(f"| {label} | {summary([getattr(t, key) for t in ok], nd)} |")
            lines += ["",
                      f"- Excluded: {', '.join(f'{v} {k}' for k, v in excluded.items()) or 'none'}"
                      + (" (\"no rest\": still moving 30 s after the stop)" if "no rest" in excluded else "") + ".",
                      f"- Stops replaced by a later command before rest: {over}/{len(cell)} "
                      f"[{lo:.3f}, {hi:.3f}].",
                      f"- Largest gap between ticks during a stop: "
                      f"{_fmt(max((t.tick_gap_max_ms for t in ok if math.isfinite(t.tick_gap_max_ms)), default=float('nan')))} ms."]
    for _, _, x in runs:
        hits = x["cue_hits"]
        k = sum(h["hit"] for h in hits)
        if hits:
            lo, hi = clopper_pearson(k, len(hits))
            lines.append(f"- {x['run']} ({x['condition']}): cues answered by a reflex stop before the next "
                         f"formation command: {k}/{len(hits)} [{lo:.3f}, {hi:.3f}].")
    lines += ["", "#### Reflex detection of spoken stop phrases", "",
              "A stop phrase is counted when the parse path's transcript holds the stop word (\"hold\" or "
              "\"abort\"); the reflex path detected it if it published in the 4 s before the parse path did. "
              "Phrases the recogniser garbled (\"for an hole\", \"from home\") are left out of both columns.", "",
              "| Condition | Phrase | spoken (transcribed) | reflex fired | miss rate [95% CI] |",
              "| :--- | :--- | ---: | ---: | :--- |"]
    for condition in CONDITIONS:
        for intent in INTENTS:
            ph = [p for _, _, x in runs if x["condition"] == condition for p in x["phrases"] if p["intent"] == intent]
            if not ph:
                continue
            fired = sum(p["fired"] for p in ph)
            lo, hi = clopper_pearson(len(ph) - fired, len(ph))
            word = "swarm hold" if intent == "hover" else "swarm abort"
            lines.append(f"| {condition} | {word} | {len(ph)} | {fired} | "
                         f"{(len(ph) - fired) / len(ph):.3f} [{lo:.3f}, {hi:.3f}] |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT_MD)
    ap.add_argument("--takes", type=Path, default=OUT_TAKES)
    args = ap.parse_args(argv)
    from eval.anchors import closed_loop_onsets

    onsets = closed_loop_onsets()
    runs = [analyse_run(r, onsets) for r in sorted(args.raw.iterdir()) if (r / "meta.json").is_file()]
    if not runs:
        print(f"no runs under {args.raw}", file=sys.stderr)
        return 1
    note = ("Utterance onset: the forced-aligned start of \"swarm\" (P15); the energy rule's region "
            "decides \"onset ambiguous\" (`results/forced_align_anchors.md` compares the anchors).")
    args.out.write_text(report(runs, note), encoding="utf-8")
    with open(args.takes, "w", encoding="utf-8") as fh:
        for _, takes, _ in runs:
            for t in takes:
                fh.write(json.dumps(asdict(t)) + "\n")
    print(args.out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
