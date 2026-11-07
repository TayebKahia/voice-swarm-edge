"""Keyword anchors of record: the forced-aligned start and end of the spoken stop phrase (P15).

The raw logs time every reflex and stop latency from an energy anchor: the capture tool's speech
bounds on the 40 recorded takes (`session_meta.json`) and `closed_loop.find_onset` on the
closed-loop captures. `eval/forced_align_anchors.py` aligned the same audio with the Montreal
Forced Aligner and wrote the aligned bounds to `results/forced_align_anchors.json`. The articles
quote latencies from the aligned anchors, so the generators re-state theirs through this module:

- recorded takes: add `take_shift_ms(take, kind)` to a latency measured from the energy anchor;
- closed loop: pass `closed_loop_onsets()` to `closed_loop.analyse_run`.

Onset is the start of "swarm"; offset is the end of "hold"/"abort". The energy figures stay in
`results/forced_align_anchors.md`, which compares the anchors.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ANCHORS = Path(__file__).resolve().parent.parent / "results" / "forced_align_anchors.json"
ALIGNER = "mfa"
RATE = 16_000
#: One sentence for the generated reports, so they all name the anchor the same way.
NOTE = ("the end (offset) or start (onset) of the spoken phrase from a forced alignment "
        "(Montreal Forced Aligner, P15; `results/forced_align_anchors.md` compares it with the "
        "capture tool's energy bounds)")


@lru_cache(maxsize=1)
def _record() -> dict:
    return json.loads(ANCHORS.read_text(encoding="utf-8"))


def take_shift_ms(take: str, kind: str) -> float:
    """Energy anchor minus aligned anchor of a recorded take, in ms; `kind` is "onset" or "offset".

    Adding it to a latency timed from the energy anchor gives the latency from the aligned one.
    """
    item = next(t for t in _record()["per_take"] if t["take"] == take)
    aligned = item[ALIGNER]
    if kind == "onset":
        return (item["energy_onset_s"] - aligned["swarm_onset"]) * 1000
    if kind == "offset":
        return (item["energy_offset_s"] - aligned["kw_offset"]) * 1000
    raise ValueError(kind)


def closed_loop_onsets() -> dict[tuple[str, str, int], int | None]:
    """(run, path, seq) -> capture sample of the aligned start of "swarm", or None where the
    aligner failed on that phrase (the stop then counts as "onset not found")."""
    out: dict[tuple[str, str, int], int | None] = {}
    for p in _record()["per_phrase"]:
        a = p.get(ALIGNER)
        out[(p["run"], p["path"], int(p["seq"]))] = (
            None if a is None else p["window_start"] + int(round(a["swarm_onset"] * RATE)))
    return out
