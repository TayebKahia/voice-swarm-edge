#!/usr/bin/env python3
"""Where the golden-set errors are, and how uncertain exact match is once that is known.

The golden items are paraphrases of 12 template families, and a family fails as a
unit: its items share a pattern, so a model that has not learned the pattern misses
most of them. Four questions follow, each answered from the per-item predictions:

1. **Per family.** Exact match of every model in each of the 12 families, so a reader
   can see which pattern carries the errors rather than infer it from a pooled score.
2. **Clustered intervals.** The item-level bootstrap of `eval/em_ci.py` resamples
   items; here whole families are resampled (`eval.stats.cluster_bootstrap_ci`), for
   each configuration and for the paired differences between them. Twelve clusters
   make a rough interval, so both are reported.
3. **One contrast, three splits.** The Qwen-against-Llama comparison at the deployed
   quantisation, on every split rather than on the golden set alone.
4. **Rotation phrasing in training.** The rotate family holds most errors, and every
   one is a yaw sign error. How often training says "clockwise", "left" or "right",
   and with which sign, is the first thing to check before explaining it any other way.

Correctness comes from `eval.metrics` through the loaders in `eval.mcnemar`, unchanged.

Usage:
    python eval/error_families.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):  # `python eval/error_families.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.mcnemar import (MODELS, QUANTS, SPLITS, _quiet_validate_logging, load_correct,
                          load_reference, paired)
from eval.stats import bootstrap_ratio_ci, cluster_bootstrap_ci, mcnemar

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "data" / "test_golden.jsonl"
TRAIN = REPO / "data" / "train.jsonl"
OUT_MD = REPO / "results" / "error_families.md"

SPLIT = "test_golden"
DEPLOYED = "Q4_K_M"
CONTROL = "h2o-danube3-500m-chat"
SHORT = {"llama-3.2-1b-instruct": "Llama", "qwen2.5-0.5b-instruct": "Qwen",
         "smollm2-360m-instruct": "SmolLM2", CONTROL: "Danube3"}

#: Direction cues in a rotate transcript, checked in this order. An explicit "minus"
#: decides the sign outright; "counter-clockwise" must be tested before "clockwise";
#: "left" before "right", because "right," also opens sentences as a discourse marker.
CUES = (("minus", re.compile(r"\bminus\b|(?<![\w.])-\d")),
        ("counter-clockwise", re.compile(r"counter[- ]?clockwise|anti[- ]?clockwise")),
        ("clockwise", re.compile(r"clockwise")),
        ("left", re.compile(r"\bleft\b")),
        ("right", re.compile(r"\bright\b")))


def cue(transcript: str) -> str:
    text = transcript.lower()
    return next((name for name, pattern in CUES if pattern.search(text)), "none")


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def families() -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for row in _rows(GOLDEN):
        out[row["template_family"]].append(row["id"])
    return dict(sorted(out.items()))


def _clusters(correct: dict[str, bool], fams: dict[str, list[str]]) -> list[tuple[float, float]]:
    return [(float(sum(correct[i] for i in ids)), float(len(ids))) for ids in fams.values()]


def _diff_clusters(first: dict, second: dict, fams: dict) -> list[tuple[float, float]]:
    return [(float(sum(first[i] - second[i] for i in ids)), float(len(ids)))
            for ids in fams.values()]


def configurations() -> dict[str, dict[str, bool]]:
    configs = {f"{SHORT[m]} {q}": load_correct(m, q, SPLIT) for m in MODELS for q in QUANTS}
    configs |= {f"{SHORT[m]} fp16": load_reference(m, SPLIT) for m in (*MODELS, CONTROL)}
    return configs


def rotation_counts(train: list[dict], golden: list[dict], wrong: dict[str, set[str]]):
    """cue -> (training pairs with positive yaw, with negative yaw); and per golden cue,
    items and errors per model."""
    in_train: dict[str, Counter] = defaultdict(Counter)
    for row in train:
        target = json.loads(row["target"])
        if target.get("intent") == "rotate" and "yaw" in target:
            in_train[cue(row["transcript"])]["neg" if target["yaw"] < 0 else "pos"] += 1
    in_golden: dict[str, Counter] = defaultdict(Counter)
    for row in golden:
        target = json.loads(row["target"])
        if target.get("intent") != "rotate":
            continue
        name = cue(row["transcript"])
        in_golden[name]["items"] += 1
        in_golden[name]["neg" if target["yaw"] < 0 else "pos"] += 1
        for label, ids in wrong.items():
            in_golden[name][label] += row["id"] in ids
    return in_train, in_golden


def markdown() -> str:
    _quiet_validate_logging()
    fams = families()
    golden = _rows(GOLDEN)
    intent = {r["template_family"]: r["gold_intent"] for r in golden}
    configs = configurations()
    rotate_family = next(f for f, i in intent.items() if i == "rotate")

    lines = ["### Golden-set errors by template family", "",
             "Per-item predictions, scored by `eval/metrics.py`. Deployed rows: `llama.cpp`,",
             "grammar on. fp16 rows: the fine-tuned reference model under `transformers`.",
             f"{len(golden)} items in {len(fams)} template families.", "",
             "#### Exact match with item-level and family-clustered intervals", "",
             "Item-level: items resampled (as `results/em_ci.md`). Clustered: the 12 families",
             "resampled whole. Both percentile bootstraps, 10,000 replicates, fixed seed.",
             f"`without {rotate_family}`: the {len(golden) - len(fams[rotate_family])} items "
             f"outside the rotate family.", "",
             "| Configuration | EM | 95% CI, items | 95% CI, families | EM without "
             f"{rotate_family} |", "| :--- | ---: | ---: | ---: | ---: |"]
    for label, correct in configs.items():
        hits = [1.0 if correct[i] else 0.0 for i in sorted(correct)]
        i_low, i_high = bootstrap_ratio_ci(hits, [1.0] * len(hits))
        c_low, c_high = cluster_bootstrap_ci(_clusters(correct, fams))
        rest = [i for f, ids in fams.items() if f != rotate_family for i in ids]
        lines.append(f"| {label} | {sum(hits) / len(hits):.3f} | [{i_low:.3f}, {i_high:.3f}] | "
                     f"[{c_low:.3f}, {c_high:.3f}] | "
                     f"{sum(correct[i] for i in rest) / len(rest):.3f} |")

    names = list(configs)
    shown = [f"{SHORT[m]} {DEPLOYED}" for m in MODELS] + [f"{SHORT[m]} fp16"
                                                          for m in ("qwen2.5-0.5b-instruct", CONTROL)]
    lines += ["", "#### Correct items per family", "",
              "| Family | Intent | n | " + " | ".join(shown) + " |",
              "| :--- | :--- | ---: | " + " | ".join("---:" for _ in shown) + " |"]
    for fam, ids in fams.items():
        lines.append(f"| {fam} | {intent[fam]} | {len(ids)} | "
                     + " | ".join(str(sum(configs[c][i] for i in ids)) for c in shown) + " |")
    lines.append("| all | | " + str(len(golden)) + " | "
                 + " | ".join(str(sum(configs[c].values())) for c in shown) + " |")

    pairs = [(f"Qwen {DEPLOYED}", f"Llama {DEPLOYED}"), (f"Qwen {DEPLOYED}", f"SmolLM2 {DEPLOYED}"),
             (f"Llama {DEPLOYED}", f"SmolLM2 {DEPLOYED}"), ("Qwen fp16", "Danube3 fp16")]
    pairs += [(f"{SHORT[m]} Q4_K_M", f"{SHORT[m]} Q8_0") for m in MODELS]
    lines += ["", "#### Paired differences in exact match", "",
              "A minus B on the same items. b / c: items only A / only B gets right; p: exact",
              "McNemar (item-level). Clustered interval: families resampled whole.", "",
              "| A | B | A - B (pp) | b / c | exact p | 95% CI, families (pp) |",
              "| :--- | :--- | ---: | ---: | ---: | ---: |"]
    for a, b in pairs:
        first, second = configs[a], configs[b]
        x, y = paired(first, second)
        result = mcnemar(x, y)
        low, high = cluster_bootstrap_ci(_diff_clusters(first, second, fams))
        delta = (sum(x) - sum(y)) / len(x)
        lines.append(f"| {a} | {b} | {100 * delta:+.1f} | {result.only_first} / "
                     f"{result.only_second} | {result.p_exact:.3g} | "
                     f"[{100 * low:+.1f}, {100 * high:+.1f}] |")

    lines += ["", f"#### Qwen against Llama at {DEPLOYED} on every split", "",
              "| Split | n | EM Qwen | EM Llama | only Qwen / only Llama | exact p |",
              "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for split in SPLITS:
        x, y = paired(load_correct("qwen2.5-0.5b-instruct", DEPLOYED, split),
                      load_correct("llama-3.2-1b-instruct", DEPLOYED, split))
        result = mcnemar(x, y)
        lines.append(f"| `{split}` | {len(x)} | {sum(x) / len(x):.3f} | {sum(y) / len(y):.3f} | "
                     f"{result.only_first} / {result.only_second} | {result.p_exact:.3g} |")

    wrong = {SHORT[m]: {i for i, ok in configs[f"{SHORT[m]} {DEPLOYED}"].items() if not ok}
             for m in MODELS}
    in_train, in_golden = rotation_counts(_rows(TRAIN), golden, wrong)
    total = sum(sum(c.values()) for c in in_train.values())
    lines += ["", "#### Direction phrasing of rotate commands", "",
              "Cue: the first match, in order, of an explicit minus, counter-clockwise,",
              "clockwise, left, right. The schema encodes counter-clockwise as a positive yaw.",
              "", f"Training (`data/train.jsonl`, {total} rotate pairs):", "",
              "| Cue | yaw > 0 | yaw < 0 |", "| :--- | ---: | ---: |"]
    for name, _ in (*CUES, ("none", None)):
        counts = in_train.get(name, Counter())
        lines.append(f"| {name} | {counts['pos']} | {counts['neg']} |")
    lines += ["", f"Golden rotate items, and errors at {DEPLOYED}:", "",
              "| Cue | items | yaw < 0 | " + " | ".join(f"{s} wrong" for s in wrong) + " |",
              "| :--- | ---: | ---: | " + " | ".join("---:" for _ in wrong) + " |"]
    for name, _ in (*CUES, ("none", None)):
        counts = in_golden.get(name)
        if counts:
            lines.append(f"| {name} | {counts['items']} | {counts['neg']} | "
                         + " | ".join(str(counts[s]) for s in wrong) + " |")
    return "\n".join(lines) + "\n"


def run() -> int:
    text = markdown()
    OUT_MD.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
