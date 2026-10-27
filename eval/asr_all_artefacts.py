#!/usr/bin/env python3
"""Every quantised artefact on the Whisper transcripts Exp-3 already recorded (A-M9i).

The master article ranks six artefacts on reference text and carries only one of them,
`qwen2.5-0.5b-instruct-Q4_K_M`, through speech recognition (Table 20). Whether the
ranking survives the ASR stage is therefore unknown: a larger model could recover more
of Whisper's errors, or a smaller one could lose more. This script answers that with no
new audio work. Exp-3's Pi run (`results/exp3_pi_preds/`) stored every endpointed
segment's transcript; each artefact parses those same transcripts here, so the
artefacts differ in the language-model stage only.

**Scoring is Exp-3's, unchanged.** `eval.exp3.score_segment` per segment and
`eval.exp3.outcome` per item: an item is recognised only if it produced at least one
segment and every command it dispatched was correct, and a failed item is safe only if
it dispatched nothing other than `unknown` or `hover`. The `text` condition (reference
transcripts) gives each artefact's EM in the same run, so EM - CRR is its ASR cost.

**Decoding is the runtime's.** `runtime.parser.CommandParser` under `schema/cmd.gbnf`,
greedy, one slot, no prompt cache. Accuracy is a property of the weights, grammar and
prompt (`eval/surface_b.py`'s docstring), so this runs on the workstation. As a check,
the Qwen Q4_K_M rows are compared with the Pi's per-segment output for the same
transcripts: any difference is an x86/ARM decoding difference and is reported.

Intervals are family-clustered bootstrap over the 12 template families
(`eval.stats.cluster_bootstrap_ci`), as in the master article's Table 1. Paired
comparisons against the deployed artefact are exact McNemar on the same items.

Usage:
    python eval/asr_all_artefacts.py                 # run what is missing, then analyse
    python eval/asr_all_artefacts.py --analyse-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):  # `python eval/asr_all_artefacts.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.error_families import families
from eval.exp3 import outcome, score_segment
from eval.stats import cluster_bootstrap_ci, mcnemar

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "results" / "exp3_pi_preds"
PREDS = REPO / "results" / "asr_all_artefacts_preds"
OUT_MD = REPO / "results" / "asr_all_artefacts.md"
GGUF_DIR = REPO / "gguf"
PORT = 8917  # clear of surface_b (8912), parser (8914), exp3 (8915), exp2 (8916)

ARTEFACTS = (
    "llama-3.2-1b-instruct-Q8_0", "llama-3.2-1b-instruct-Q4_K_M",
    "qwen2.5-0.5b-instruct-Q8_0", "qwen2.5-0.5b-instruct-Q4_K_M",
    "smollm2-360m-instruct-Q8_0", "smollm2-360m-instruct-Q4_K_M",
)
DEPLOYED = "qwen2.5-0.5b-instruct-Q4_K_M"
CONDITIONS = ("text", "clean", "20", "15", "10", "5")


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_artefact(stem: str, fresh: bool) -> None:
    pending = [c for c in CONDITIONS if fresh or not (PREDS / stem / f"{c}.jsonl").is_file()]
    if not pending:
        print(f"{stem}: complete")
        return
    from runtime.parser import CommandParser

    parser, server = CommandParser.for_gguf(GGUF_DIR / f"{stem}.gguf", port=PORT)
    try:
        for condition in pending:
            rows = []
            for source in _read(SOURCE / f"{condition}.jsonl"):
                segments = []
                for seg in source["segments"]:
                    raw = parser.parse(seg["transcript"]).raw
                    segments.append({"transcript": seg["transcript"], "raw": raw,
                                     **score_segment(raw, source["gold"])})
                rows.append({"id": source["id"], "condition": condition, "gold": source["gold"],
                             "segments": segments})
            _write(PREDS / stem / f"{condition}.jsonl", rows)
            print(f"  {stem:32s} {condition:>5s}: {len(rows)} items", flush=True)
    finally:
        server.stop()


# --- analysis ----------------------------------------------------------------


def _outcomes(stem: str, condition: str) -> dict[str, object]:
    return {r["id"]: outcome(r["segments"]) for r in _read(PREDS / stem / f"{condition}.jsonl")}


def _ci(correct: dict[str, bool], fams: dict[str, list[str]]) -> tuple[float, float]:
    return cluster_bootstrap_ci([(float(sum(correct[i] for i in ids)), float(len(ids)))
                                 for ids in fams.values()])


def parity_with_pi() -> list[str]:
    """Deployed artefact here against the Pi's own decode of the same transcripts."""
    lines = []
    for condition in CONDITIONS:
        mine = {r["id"]: r for r in _read(PREDS / DEPLOYED / f"{condition}.jsonl")}
        raw_diff = flips = 0
        for pi in _read(SOURCE / f"{condition}.jsonl"):
            here = mine[pi["id"]]
            raw_diff += sum(a["raw"] != b["raw"] for a, b in zip(pi["segments"], here["segments"]))
            flips += outcome(pi["segments"]).correct != outcome(here["segments"]).correct
        lines.append(f"| {condition} | {raw_diff} | {flips} |")
    return lines


def analyse() -> int:
    fams = families()
    ids = [i for ids in fams.values() for i in ids]
    table, by = [], defaultdict(dict)
    for stem in ARTEFACTS:
        em = {i: o.correct for i, o in _outcomes(stem, "text").items()}
        for condition in CONDITIONS[1:]:
            outs = _outcomes(stem, condition)
            crr = {i: o.correct for i, o in outs.items()}
            by[condition][stem] = crr
            failed = [o for o in outs.values() if not o.correct]
            lo, hi = _ci(crr, fams)
            table.append({"artefact": stem, "condition": condition,
                          "em": sum(em.values()) / len(ids), "crr": sum(crr.values()) / len(ids),
                          "lo": lo, "hi": hi, "safe": sum(o.safe for o in failed),
                          "failed": len(failed)})

    lines = [
        "### Every artefact on the recorded Whisper transcripts (A-M9i)",
        "",
        "Generated by `python eval/asr_all_artefacts.py`. Do not edit by hand.",
        "",
        "Transcripts: Exp-3's Pi run (`results/exp3_pi_preds/`; whisper `tiny.en`, domain "
        "prompt, Silero endpointing at 450 ms, DREGON rotor noise mixed on active-speech "
        "level). Each artefact parses the same per-segment transcripts under the grammar; "
        "scoring is Exp-3's rule of record. n = 200 golden items; 95% intervals are "
        "family-clustered bootstrap over the 12 template families. EM is the same "
        "artefact on the reference transcripts in the same run.",
        "",
        "| Artefact | SNR | EM | CRR | 95% CI | EM - CRR | Safe failures |",
        "| :--- | :--- | ---: | ---: | :--- | ---: | :--- |",
    ]
    for r in table:
        label = "clean" if r["condition"] == "clean" else f"{r['condition']} dB"
        lines.append(f"| `{r['artefact']}` | {label} | {r['em']:.3f} | {r['crr']:.3f} "
                     f"| [{r['lo']:.3f}, {r['hi']:.3f}] | {r['em'] - r['crr']:+.3f} "
                     f"| {r['safe']}/{r['failed']} |")

    lines += ["", f"#### Paired against the deployed artefact (`{DEPLOYED}`), exact McNemar",
              "", "b: the other artefact right and the deployed one wrong; c: the reverse.", "",
              "| Artefact | SNR | b | c | p (exact) |", "| :--- | :--- | ---: | ---: | ---: |"]
    for condition in CONDITIONS[1:]:
        ref = by[condition][DEPLOYED]
        for stem in ARTEFACTS:
            if stem == DEPLOYED:
                continue
            result = mcnemar([by[condition][stem][i] for i in ids], [ref[i] for i in ids])
            label = "clean" if condition == "clean" else f"{condition} dB"
            lines.append(f"| `{stem}` | {label} | {result.only_first} | {result.only_second} "
                         f"| {result.p_exact:.3g} |")

    lines += ["", "#### Check: deployed artefact here against the Pi's decode",
              "", "Same transcripts in, so a difference is x86 against ARM decoding.", "",
              "| Condition | segments with different raw output | item outcome flips |",
              "| :--- | ---: | ---: |", *parity_with_pi(), "",
              "No multiple-comparison correction is applied to the paired table; it is "
              "descriptive, and a reader testing the ranking should take Bonferroni over its "
              "25 rows (alpha = 0.002)."]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    for r in table:
        if r["condition"] == "clean":
            print(f"  {r['artefact']:32s} EM {r['em']:.3f}  CRR {r['crr']:.3f} "
                  f"[{r['lo']:.3f}, {r['hi']:.3f}]")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--artefacts", nargs="+", default=list(ARTEFACTS), choices=ARTEFACTS)
    args = parser.parse_args(argv)
    if not args.analyse_only:
        for stem in args.artefacts:
            run_artefact(stem, args.fresh)
    return analyse()


if __name__ == "__main__":
    sys.exit(main())
