#!/usr/bin/env python3
"""95% bootstrap intervals for exact match, re-derived from the per-item predictions.

The model comparison reports exact match as point estimates with paired McNemar tests;
this script adds the interval a reader needs to judge one figure on its own. It reuses
the two definitions of record unchanged: correctness comes from `eval.metrics` through
the loaders in `eval.mcnemar`, and the interval from `eval.stats.bootstrap_ratio_ci`
(percentile bootstrap, 10,000 replicates, fixed seed, nearest-rank bounds).

**The resampling unit is the item.** The golden items are paraphrases of 12 command
patterns, so they are not independent, and an item-level interval is narrower than
the uncertainty it describes --- the same caveat the McNemar tests carry. A cluster
bootstrap over 12 patterns would be unreliable in its own way, so the item-level
interval is reported and the caveat is stated beside it.

Every point estimate is checked against `results/surface_b.csv` (deployed surface) or
`train/kaggle_out/surface_a.csv` (reference surface) before anything is written.

Usage:
    python eval/em_ci.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python eval/em_ci.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.mcnemar import (MODELS, QUANTS, SPLITS, SURFACE_A, SURFACE_B, _display,
                          _quiet_validate_logging, load_correct, load_reference)
from eval.stats import BOOTSTRAP_RESAMPLES, bootstrap_ratio_ci

REPO = Path(__file__).resolve().parent.parent
OUT_CSV = REPO / "results" / "em_ci.csv"
OUT_MD = REPO / "results" / "em_ci.md"

#: The reference surface adds the parameter-matched control to the three candidates.
REFERENCE_MODELS = (*MODELS, "h2o-danube3-500m-chat")

FIELDNAMES = ["surface", "model", "quant", "split", "n", "exact_match", "ci_low", "ci_high"]


def _row(surface: str, model: str, quant: str, split: str, correct: dict[str, bool]) -> dict:
    hits = [1.0 if correct[i] else 0.0 for i in sorted(correct)]
    low, high = bootstrap_ratio_ci(hits, [1.0] * len(hits))
    return {"surface": surface, "model": model, "quant": quant, "split": split,
            "n": len(hits), "exact_match": sum(hits) / len(hits),
            "ci_low": low, "ci_high": high}


def _recorded(path: Path, keep) -> dict[tuple, float]:
    with path.open(encoding="utf-8") as handle:
        return {key: float(r["exact_match"]) for r in csv.DictReader(handle)
                if (key := keep(r)) is not None}


def rows() -> list[dict]:
    _quiet_validate_logging()
    out = [_row("deployed", m, q, s, load_correct(m, q, s))
           for m in MODELS for q in QUANTS for s in SPLITS]
    out += [_row("reference", m, "fp16", s, load_reference(m, s))
            for m in REFERENCE_MODELS for s in SPLITS]
    return out


def check(computed: list[dict]) -> list[str]:
    deployed = _recorded(SURFACE_B, lambda r: (r["model"], r["quant"], r["split"])
                         if r["grammar"] == "on" else None)
    reference = _recorded(SURFACE_A, lambda r: (r["model"], "fp16", r["split"])
                          if r["surface"] == "A_fp16_finetuned" else None)
    problems = []
    for row in computed:
        recorded = (deployed if row["surface"] == "deployed" else reference).get(
            (row["model"], row["quant"], row["split"]))
        if recorded is None or abs(recorded - row["exact_match"]) > 5e-4:
            problems.append(f"{row['surface']} {row['model']} {row['quant']} {row['split']}: "
                            f"re-derived {row['exact_match']:.4f}, recorded {recorded}")
    return problems


def markdown(computed: list[dict]) -> str:
    lines = [
        "### Exact match with 95% bootstrap intervals",
        "",
        "Re-derived from the per-item predictions (`results/surface_b_preds/`, grammar on;",
        "`train/kaggle_out/preds_*.jsonl` for the fp16 reference) by `eval/em_ci.py`.",
        f"Percentile bootstrap over items, {BOOTSTRAP_RESAMPLES:,} replicates, fixed seed,",
        "nearest-rank bounds. The golden items are paraphrases of 12 command patterns, so",
        "the item-level interval is narrower than the true uncertainty.",
        "",
        "| Surface | Model | Quant | Split | n | EM | 95% CI |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | :--- |",
    ]
    for r in computed:
        lines.append(f"| {r['surface']} | {r['model']} | {r['quant']} | `{r['split']}` | "
                     f"{r['n']} | {r['exact_match']:.3f} | "
                     f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] |")
    return "\n".join(lines) + "\n"


def run() -> int:
    computed = rows()
    problems = check(computed)
    if problems:
        print("\n".join(problems))
        raise SystemExit("re-derived exact match disagrees with the recorded sweep")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in computed:
            writer.writerow({**r, "exact_match": f"{r['exact_match']:.4f}",
                             "ci_low": f"{r['ci_low']:.4f}", "ci_high": f"{r['ci_high']:.4f}"})
    OUT_MD.write_text(markdown(computed), encoding="utf-8")
    print(f"wrote {_display(OUT_CSV)} and {_display(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
