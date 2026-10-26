#!/usr/bin/env python3
"""Two breakdowns that pooled figures hide: out-of-domain sources and the grammar per model.

**Out-of-domain set by source.** `test_ood` mixes four sources in near-equal shares, and
they are not equally hard: an assistant query shares no words with the schema, while a
truncated fragment ("move north") is a command missing its slots. The pooled
false-command rate therefore depends on the mix the authors chose. This script reports
it per source and states who wrote each item (`provenance` in `data/test_ood.jsonl`).

**Grammar ablation per model.** Table 19 pools 3,540 decodes, but they are the same 590
items decoded by six correlated artefacts, so the pooled count is not 3,540 independent
chances to go wrong. Per artefact: malformed outputs with and without the grammar, an
exact (Clopper-Pearson) upper bound on the malformed rate --- with zero events this is
close to the rule of three, 3/n --- and an exact McNemar test on exact match, grammar
on against grammar off, over the same items.

Usage:
    python eval/ood_and_grammar.py
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):  # `python eval/ood_and_grammar.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import binomtest

from eval.mcnemar import MODELS, QUANTS, SPLITS, _quiet_validate_logging
from eval.metrics import score_predictions
from eval.stats import mcnemar

REPO = Path(__file__).resolve().parent.parent
PREDICTIONS = REPO / "results" / "surface_b_preds"
OOD = REPO / "data" / "test_ood.jsonl"
OUT_MD = REPO / "results" / "ood_and_grammar.md"

SHORT = {"llama-3.2-1b-instruct": "Llama", "qwen2.5-0.5b-instruct": "Qwen",
         "smollm2-360m-instruct": "SmolLM2"}


def _predictions(model: str, quant: str, split: str, *, grammar: bool = True):
    suffix = "" if grammar else "_nogrammar"
    path = PREDICTIONS / f"{model}-{quant}{suffix}_{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return rows, score_predictions([r["raw"] for r in rows], [r["gold"] for r in rows])


def ood_section() -> list[str]:
    items = [json.loads(line) for line in OOD.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    source = {r["id"]: r["ood_source"] for r in items}
    sources = sorted(Counter(source.values()))
    provenance = Counter((r["ood_source"], r["provenance"]) for r in items)

    lines = ["### Out-of-domain set by source", "",
             f"`data/test_ood.jsonl`, {len(items)} items, every reference `unknown`.", "",
             "| Source | Provenance | Items |", "| :--- | :--- | ---: |"]
    for (src, prov), count in sorted(provenance.items()):
        lines.append(f"| {src} | {prov} | {count} |")

    lines += ["", "False-command rate (dispatched action other than `unknown`) per source,",
              "grammar on:", "",
              "| Configuration | " + " | ".join(sources) + " | all |",
              "| :--- | " + " | ".join("---:" for _ in sources) + " | ---: |"]
    for model, quant in itertools.product(MODELS, QUANTS):
        rows, preds = _predictions(model, quant, "test_ood")
        false = Counter(source[r["id"]] for r, p in zip(rows, preds) if p.dispatched != "unknown")
        total = Counter(source[r["id"]] for r in rows)
        cells = [f"{false[s]}/{total[s]}" for s in sources]
        lines.append(f"| {SHORT[model]} {quant} | " + " | ".join(cells)
                     + f" | {sum(false.values()) / len(rows):.3f} |")
    return lines


def grammar_section() -> list[str]:
    lines = ["", "### Grammar ablation per artefact", "",
             "The 590 items of the three test splits, greedy, grammar on and off; nothing else",
             "changes. Malformed: output that does not parse as a schema-valid command. Upper",
             "bound: exact one-sided 95% (Clopper-Pearson) bound on the malformed rate without",
             "the grammar. b / c: items only grammar-on / only grammar-off gets right.", "",
             "| Artefact | Malformed on | Malformed off | 95% upper bound, off | EM on | EM off "
             "| b / c | exact p |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model, quant in itertools.product(MODELS, QUANTS):
        on = [p for s in SPLITS for p in _predictions(model, quant, s)[1]]
        off = [p for s in SPLITS for p in _predictions(model, quant, s, grammar=False)[1]]
        bad_on = sum(not p.parsed_ok for p in on)
        bad_off = sum(not p.parsed_ok for p in off)
        upper = binomtest(bad_off, len(off)).proportion_ci(0.90, method="exact").high
        result = mcnemar([p.correct for p in on], [p.correct for p in off])
        lines.append(f"| {SHORT[model]} {quant} | {bad_on} | {bad_off} | {upper:.4f} | "
                     f"{sum(p.correct for p in on) / len(on):.4f} | "
                     f"{sum(p.correct for p in off) / len(off):.4f} | "
                     f"{result.only_first} / {result.only_second} | {result.p_exact:.3g} |")
    return lines


def run() -> int:
    _quiet_validate_logging()
    text = "\n".join(ood_section() + grammar_section()) + "\n"
    OUT_MD.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
