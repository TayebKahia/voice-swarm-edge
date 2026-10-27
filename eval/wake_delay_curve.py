#!/usr/bin/env python3
"""Reflex-path delay against threshold, with an exact anchor beside the energy one.

The ingenieur article's reflex path misses its 150 ms budget (p95 545 ms from keyword
offset, Exp-2), and the review (B-M2) names three things the paper never separated:
the threshold sits at 0.999, where the heads' scores saturate; the offset anchor is an
energy estimate, and 10 of 156 detections precede it; and delay against threshold was
never measured. This script measures the curve and puts a second anchor beside the
first. It retrains nothing and selects nothing: the operating point stays the one
`train/train_wake.py` chose on val.

**Three anchors.**

- *Exact (synthetic).* The test split of the keyword corpus, restricted to the clips
  whose keyword end is known to the sample (`train_wake.keyword_spans`, `exact`: the
  dry rendition's last voiced sample). Same three Piper voices as training, so this is
  the spotter's algorithmic delay with no anchor error, not its delay on a real voice.
- *Energy (real).* The 40 recorded takes, offset from the capture tool's own speech
  bounds (`session_meta.json`: 20 ms frames above max(floor + 12 dB, peak - 26 dB)).
  This is the anchor Exp-2 and `eval/wake_real.py` use.
- *Silero (real).* The same takes, offset at the end of the last 32 ms window whose
  Silero speech probability is >= 0.5 (the endpointer's own model and threshold).
  An independent detector, not a forced alignment: it shows how far the real-voice
  figure moves with the anchor, which is the question the 150 ms scale turns on.

**Delay is algorithmic.** First 80 ms frame at or above threshold, measured from the
anchor, exactly as `train_wake.score_positives` measures it. Exp-2 shows the Pi adds
5 ms idle and 13-18 ms loaded on top (`results/latency_ci.md`), so the curve here is
the part of the budget that a faster board cannot buy back.

**False accepts are the price.** Each threshold also carries the false-accept rate on
the held-out ambient stream (NFR-15), the authored near-miss stream and the author's
200 golden utterances, counted with the 1 s debounce as in training.

Percentile intervals are distribution-free (order statistics); FRR intervals are exact
(Clopper-Pearson); false-accept intervals are Poisson. The corpus audio is not in git:
`data/wake/` must hold the rendered corpus (`data/wake_corpus.py`).

Usage:
    python eval/wake_delay_curve.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python eval/wake_delay_curve.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from eval.stats import nearest_rank, order_statistic_ci

REPO = Path(__file__).resolve().parent.parent
OUT_MD = REPO / "results" / "wake_delay_curve.md"
OUT_JSON = REPO / "results" / "wake_delay_curve.json"

#: Coarse below 0.9, fine above it, where the operating point and the saturation are.
GRID = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999)
BUDGET_MS = 150.0  # Table 6, reflex path from keyword offset
VAD_THRESHOLD = 0.5


def silero_offsets(takes: dict[str, np.ndarray]) -> dict[str, float]:
    """take -> end of the last speech window, seconds from the start of the take."""
    from runtime.vad import WINDOW, SileroVAD

    vad = SileroVAD()
    out = {}
    for take, audio in takes.items():
        vad.reset()
        last = None
        for k in range(audio.size // WINDOW):
            window = audio[k * WINDOW : (k + 1) * WINDOW].astype(np.float32) / 32768.0
            if vad(window) >= VAD_THRESHOLD:
                last = k
        out[take] = math.nan if last is None else (last + 1) * WINDOW / 16_000
    return out


def delays_at(traces: list[dict], name: str, threshold: float, anchor: str) -> tuple[list[float], int]:
    """Delays (ms) of the takes of one class that fired their own head, and the miss count."""
    from train.train_wake import FRAME, RATE

    delays, missed = [], 0
    for t in traces:
        if t["class"] != name:
            continue
        above = np.flatnonzero(t["trace"][name] >= threshold)
        if above.size == 0 or math.isnan(t[anchor]):
            missed += above.size == 0
            continue
        delays.append(float(((above[0] + 1) * FRAME / RATE - t[anchor]) * 1000.0))
    return delays, missed


def percentile_cell(delays: list[float], q: float) -> dict:
    if not delays:
        return {"value": None, "lo": None, "hi": None}
    lo, hi = order_statistic_ci(delays, q)
    return {"value": nearest_rank(delays, q), "lo": None if math.isinf(lo) else lo,
            "hi": None if math.isinf(hi) else hi}


def curve(traces: list[dict], anchor: str, classes: list[str]) -> list[dict]:
    from eval.wake_real import clopper_pearson

    rows = []
    for threshold in GRID:
        pooled, per_class = [], {}
        missed_total = n_total = 0
        for name in classes:
            delays, missed = delays_at(traces, name, threshold, anchor)
            n = sum(t["class"] == name for t in traces)
            pooled += delays
            missed_total += missed
            n_total += n
            per_class[name] = {"n": n, "missed": missed,
                               "p50": percentile_cell(delays, 50)["value"],
                               "p95": percentile_cell(delays, 95)["value"]}
        lo, hi = clopper_pearson(missed_total, n_total)
        rows.append({"threshold": threshold, "n": n_total, "missed": missed_total,
                     "frr": missed_total / n_total, "frr_ci95": [lo, hi],
                     "p50": percentile_cell(pooled, 50), "p95": percentile_cell(pooled, 95),
                     "early": int(sum(d < 0 for d in pooled)),
                     "within_budget": int(sum(d <= BUDGET_MS for d in pooled)),
                     "per_class": per_class})
    return rows


def false_accepts(streams: dict[str, tuple[dict, int]]) -> list[dict]:
    from train.train_wake import score_stream

    rows = []
    for threshold in GRID:
        row = {"threshold": threshold}
        for label, (traces, samples) in streams.items():
            s = score_stream(traces, samples, threshold)
            row[label] = {"hours": float(s["hours"]), "events": int(s["total"]),
                          "per_hour": float(s["per_hour"]), "ci95": [float(v) for v in s["per_hour_ci95"]]}
        rows.append(row)
    return rows


def run() -> int:
    from eval.wake_real import META, PAD_S, TAKES, _check_heads, labels
    from train.train_wake import (CLASSES, MANIFEST, _load_int16, _model, golden_stream,
                                  keyword_spans, negative_stream, stream_scores)

    operating = _check_heads()
    model = _model()
    pad = np.zeros(int(PAD_S * 16_000), dtype=np.int16)
    names = list(CLASSES)
    label_to_name = {label: name for name, label in CLASSES.items()}

    # Real takes: two anchors on the same traces.
    meta = json.loads(META.read_text())["takes"]
    takes = {take: _load_int16(TAKES / f"{take}.wav") for take in sorted(labels())}
    silero = silero_offsets(takes)
    real = []
    for take, name in sorted(labels().items()):
        audio = takes[take]
        real.append({"take": take, "class": name,
                     "energy": PAD_S + meta[take]["seconds"] - meta[take]["tail_seconds"],
                     "silero": PAD_S + silero[take],
                     "trace": stream_scores(model, np.concatenate([pad, audio, pad]))})
    anchor_gap_ms = [(r["silero"] - r["energy"]) * 1000.0 for r in real]

    # Synthetic test split, exact keyword ends only.
    clips = json.loads(MANIFEST.read_text())["clips"]
    spans = keyword_spans(clips)
    synthetic = []
    for clip in clips:
        if clip["split"] != "test" or clip["label"] == "neg" or not spans[clip["path"]][2]:
            continue
        audio = np.concatenate([pad, _load_int16(REPO / clip["path"]), pad])
        synthetic.append({"class": label_to_name[clip["label"]],
                          "exact": PAD_S + spans[clip["path"]][1],
                          "trace": stream_scores(model, audio)})
    print(f"traces: {len(real)} real takes, {len(synthetic)} exact synthetic clips", flush=True)

    streams = {}
    for label, split, ambient in (("ambient", "test", True), ("near_miss", "test", False)):
        audio = negative_stream(clips, split, ambient)
        streams[label] = (stream_scores(model, audio), audio.size)
    golden = golden_stream()
    streams["golden"] = (stream_scores(model, golden), golden.size)

    record = {
        "operating_point": operating, "grid": list(GRID), "budget_ms": BUDGET_MS,
        "real_energy": curve(real, "energy", names),
        "real_silero": curve(real, "silero", names),
        "synthetic_exact": curve(synthetic, "exact", names),
        "false_accepts": false_accepts(streams),
        "anchor_gap_ms": {"median": float(np.median(anchor_gap_ms)),
                          "min": float(np.min(anchor_gap_ms)), "max": float(np.max(anchor_gap_ms))},
        "peaks": {r["take"]: round(float(r["trace"][r["class"]].max()), 4) for r in real},
    }
    OUT_JSON.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(markdown(record, len(synthetic)), encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(REPO)} and {OUT_JSON.relative_to(REPO)}")
    return 0


# --- report ------------------------------------------------------------------


def _ms(cell: dict) -> str:
    if cell["value"] is None:
        return "--"
    lo = "--" if cell["lo"] is None else f"{cell['lo']:.0f}"
    hi = "--" if cell["hi"] is None else f"{cell['hi']:.0f}"
    return f"{cell['value']:.0f} [{lo}, {hi}]"


def _curve_table(rows: list[dict], operating: float) -> list[str]:
    lines = ["| threshold | n | missed | FRR [95% CI] | p50 [95% CI] (ms) | p95 [95% CI] (ms) "
             "| before anchor | <= 150 ms |",
             "| ---: | ---: | ---: | :--- | :--- | :--- | ---: | ---: |"]
    for r in rows:
        mark = " (operating)" if r["threshold"] == operating else ""
        detected = r["n"] - r["missed"]
        lines.append(
            f"| {r['threshold']}{mark} | {r['n']} | {r['missed']} | {r['frr']:.3f} "
            f"[{r['frr_ci95'][0]:.3f}, {r['frr_ci95'][1]:.3f}] | {_ms(r['p50'])} | {_ms(r['p95'])} "
            f"| {r['early']}/{detected} | {r['within_budget']}/{detected} |")
    return lines


def markdown(record: dict, n_exact: int) -> str:
    op = record["operating_point"]
    gap = record["anchor_gap_ms"]
    lines = [
        "# Reflex-path delay against threshold (B-M2)",
        "",
        "Generated by `python eval/wake_delay_curve.py`. Do not edit by hand.",
        "",
        "Delay is the first 80 ms frame at or above threshold, measured from the keyword's end "
        "under the stated anchor; it is the algorithmic part of the reflex budget (150 ms from "
        "offset). The Pi adds 5 ms idle and 13-18 ms loaded (`results/latency_ci.md`). Nothing "
        f"here was used to choose a threshold; the operating point stays **{op}**, selected on "
        "val by `train/train_wake.py`. Pooled over both classes; per-class figures are in "
        "`results/wake_delay_curve.json`.",
        "",
        f"## Exact anchor: synthetic test split ({n_exact} clips, end sample known)",
        "",
        *_curve_table(record["synthetic_exact"], op),
        "",
        "## Energy anchor: the 40 real takes (capture tool's speech bounds; Exp-2's anchor)",
        "",
        *_curve_table(record["real_energy"], op),
        "",
        "## Silero anchor: the same 40 takes (last 32 ms window with p(speech) >= 0.5)",
        "",
        *_curve_table(record["real_silero"], op),
        "",
        f"Silero's end minus the energy end, per take: median {gap['median']:+.0f} ms, "
        f"range [{gap['min']:+.0f}, {gap['max']:+.0f}] ms.",
        "",
        "## False accepts per hour at each threshold (debounce 1.0 s, Poisson 95% CI)",
        "",
        "| threshold | ambient (NFR-15) | near-miss (authored) | golden (author) |",
        "| ---: | :--- | :--- | :--- |",
    ]
    for r in record["false_accepts"]:
        cells = [f"{r[k]['events']} ev, {r[k]['per_hour']:.1f}/h [{r[k]['ci95'][0]:.1f}, "
                 f"{r[k]['ci95'][1]:.1f}]" for k in ("ambient", "near_miss", "golden")]
        lines.append(f"| {r['threshold']} | " + " | ".join(cells) + " |")
    fa = record["false_accepts"][0]
    lines += [
        "",
        f"Stream lengths: ambient {fa['ambient']['hours']:.3f} h, near-miss "
        f"{fa['near_miss']['hours']:.3f} h, golden {fa['golden']['hours']:.3f} h.",
        "",
        "## Read before quoting",
        "",
        "- **The exact anchor is synthetic.** Its clips are Piper renditions in the three "
        "training voices; it bounds what the spotter does when the anchor is right, not what "
        "it does on the operator's voice.",
        "- **The real-voice anchors are detectors, not alignments.** They disagree by the "
        "amounts above; a delay difference smaller than that disagreement is not resolved.",
        "- **The curve is descriptive.** Choosing a new threshold from it would use the real "
        "takes for selection; any change of operating point must be chosen on val and "
        "re-measured on fresh takes.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(run())
