#!/usr/bin/env python3
"""S1 (review A-M4b): the rotation-balanced retrain against its control and the deployed models.

    python eval/s1_rotation.py            # writes results/s1/s1_rotation.md

Three versions of each of the two leading models, all decoded by `llama.cpp` under the
grammar (`eval/surface_b.py`), at both deployed quantisations:

    deployed   gguf/<model>-<quant>         trained on data/train.jsonl (the artefacts of record)
    ctrl       gguf_s1/<model>-ctrl-<quant> retrained on data/train.jsonl in the S1 session
    s1         gguf_s1/<model>-s1-<quant>   retrained on data/train_s1_balanced.jsonl

`s1 - ctrl` isolates the data change: both runs come from one Kaggle session, image
and library versions (transformers 5, where the deployed runs had 4). `s1 - deployed`
is the comparison against the artefacts of record, and mixes the data change with
rerun noise; `ctrl - deployed` measures that noise on its own.

Scoring is the scoring of record. Correctness comes from `eval.metrics.score_predictions`
on the raw outputs. Intervals are percentile bootstraps (10,000 replicates, the seed
`eval.stats` fixes); on `test_golden` and `test_synth` whole template families are
resampled, while on `test_ood`, which is one family, items are resampled and labelled so.
p-values are exact McNemar at item level. The direction cue of a rotate item is
`eval.error_families.cue`, the classifier the article's P9 counts use.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):  # `python eval/s1_rotation.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.error_families import CUES, cue
from eval.mcnemar import _quiet_validate_logging
from eval.metrics import score_predictions
from eval.stats import bootstrap_ratio_ci, cluster_bootstrap_ci, mcnemar

REPO = Path(__file__).resolve().parent.parent
DEPLOYED_PREDS = REPO / "results" / "surface_b_preds"
S1_PREDS = REPO / "results" / "s1" / "surface_b_preds"
OUT_MD = REPO / "results" / "s1" / "s1_rotation.md"
TRAIN = {"deployed": REPO / "data" / "train.jsonl",
         "S1 balanced": REPO / "data" / "train_s1_balanced.jsonl"}

MODELS = {"qwen2.5-0.5b-instruct": "Qwen", "llama-3.2-1b-instruct": "Llama"}
QUANTS = ("Q4_K_M", "Q8_0")
DEPLOYED_QUANT = "Q4_K_M"
SPLITS = ("test_golden", "test_synth", "test_ood")
VARIANTS = ("deployed", "ctrl", "s1")
IN_DOMAIN = ("test_golden", "test_synth")


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def pred_path(model: str, variant: str, quant: str, split: str) -> Path:
    if variant == "deployed":
        return DEPLOYED_PREDS / f"{model}-{quant}_{split}.jsonl"
    return S1_PREDS / f"{model}-{variant}-{quant}_{split}.jsonl"


def load(model: str, variant: str, quant: str, split: str) -> dict[str, dict]:
    """item id -> {correct, raw, gold}, scored by `eval.metrics`."""
    path = pred_path(model, variant, quant, split)
    if not path.is_file():
        raise SystemExit(f"missing {path} -- run eval/surface_b.py")
    rows = _rows(path)
    scored = score_predictions([r["raw"] for r in rows], [r["gold"] for r in rows])
    return {r["id"]: {"correct": p.correct, "raw": r["raw"], "gold": r["gold"]}
            for r, p in zip(rows, scored)}


def split_rows(split: str) -> list[dict]:
    return _rows(REPO / "data" / f"{split}.jsonl")


def clusters_of(split: str) -> dict[str, list[str]] | None:
    """template family -> item ids; None when the split is a single family (test_ood)."""
    out: dict[str, list[str]] = defaultdict(list)
    for row in split_rows(split):
        out[row["template_family"]].append(row["id"])
    return dict(out) if len(out) > 1 else None


def em_ci(correct: dict[str, bool], fams) -> tuple[float, float, str]:
    if fams is None:
        hits = [1.0 if correct[i] else 0.0 for i in sorted(correct)]
        low, high = bootstrap_ratio_ci(hits, [1.0] * len(hits))
        return low, high, "items"
    low, high = cluster_bootstrap_ci(
        [(float(sum(correct[i] for i in ids)), float(len(ids))) for ids in fams.values()])
    return low, high, "families"


def diff_ci(first: dict[str, bool], second: dict[str, bool], fams) -> tuple[float, float]:
    if fams is None:
        diffs = [float(first[i]) - float(second[i]) for i in sorted(first)]
        return bootstrap_ratio_ci(diffs, [1.0] * len(diffs))
    return cluster_bootstrap_ci(
        [(float(sum(first[i] - second[i] for i in ids)), float(len(ids))) for ids in fams.values()])


def _yaw(text: str) -> float | None:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(obj, dict) and isinstance(obj.get("yaw"), (int, float)):
        return float(obj["yaw"])
    return None


def is_sign_error(record: dict) -> bool:
    """Wrong only in the sign of the yaw: the swarm would turn the requested angle the other way."""
    gold, pred = _yaw(record["gold"]), _yaw(record["raw"])
    if gold is None or pred is None or gold == 0 or record["correct"]:
        return False
    if abs(pred + gold) > 0.05:
        return False
    a, b = json.loads(record["gold"]), json.loads(record["raw"])
    a.pop("yaw"), b.pop("yaw")
    return a == b


def markdown() -> str:
    _quiet_validate_logging()
    data = {(m, v, q, s): load(m, v, q, s)
            for m in MODELS for v in VARIANTS for q in QUANTS for s in SPLITS}
    fams = {s: clusters_of(s) for s in SPLITS}

    def correct(m, v, q, s):
        return {i: r["correct"] for i, r in data[(m, v, q, s)].items()}

    lines = ["### S1: rotation-balanced retrain (review A-M4b)", "",
             "`deployed`: the artefacts of record. `ctrl`: retrained on the deployed data in the",
             "S1 session. `s1`: retrained in the same session on `data/train_s1_balanced.jsonl`",
             "(19 rotation rows mirrored; `data/s1_balance_rotation.py`). Same configs and seed.",
             "All decoded by `llama.cpp` under the grammar.", ""]

    # --- training counts -----------------------------------------------------------
    lines += ["#### Rotation pairs in training, by direction cue", "",
              "| Cue | deployed yaw > 0 | deployed yaw < 0 | S1 yaw > 0 | S1 yaw < 0 |",
              "| :--- | ---: | ---: | ---: | ---: |"]
    counts = {}
    for label, path in TRAIN.items():
        c: dict[str, Counter] = defaultdict(Counter)
        for row in _rows(path):
            target = json.loads(row["target"])
            if target.get("intent") == "rotate" and "yaw" in target:
                c[cue(row["transcript"])]["neg" if target["yaw"] < 0 else "pos"] += 1
        counts[label] = c
    for name in (*(n for n, _ in CUES), "none"):
        d, s = counts["deployed"][name], counts["S1 balanced"][name]
        lines.append(f"| {name} | {d['pos']} | {d['neg']} | {s['pos']} | {s['neg']} |")

    # --- exact match -----------------------------------------------------------------
    lines += ["", "#### Exact match", "",
              "95% percentile bootstrap, 10,000 replicates; template families resampled whole on",
              "`test_golden` (12) and `test_synth` (12), items on `test_ood` (one family).", "",
              "| Model | Quant | Variant | " + " | ".join(f"`{s}` EM [95% CI]" for s in SPLITS) + " |",
              "| :--- | :--- | :--- | " + " | ".join("---:" for _ in SPLITS) + " |"]
    for m, short in MODELS.items():
        for q in QUANTS:
            for v in VARIANTS:
                cells = []
                for s in SPLITS:
                    cor = correct(m, v, q, s)
                    low, high, unit = em_ci(cor, fams[s])
                    em = sum(cor.values()) / len(cor)
                    cells.append(f"{em:.3f} [{low:.3f}, {high:.3f}]{' (items)' if unit == 'items' else ''}")
                lines.append(f"| {short} | {q} | {v} | " + " | ".join(cells) + " |")

    # --- paired tests ----------------------------------------------------------------
    pairs = (("s1", "ctrl", "the data change"), ("s1", "deployed", "against the artefacts of record"),
             ("ctrl", "deployed", "rerun noise"))
    lines += ["", "#### Paired comparisons within each model", "",
              "A minus B on the same items. b / c: items only A / only B gets right. p: exact",
              "McNemar, item level (items of one family are not independent; read with the",
              "clustered interval).", "",
              "| Model | Quant | Split | A | B | reads as | A - B (pp) | b / c | exact p | 95% CI (pp) |",
              "| :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |"]
    for m, short in MODELS.items():
        for q in QUANTS:
            for s in SPLITS:
                for a, b, reads in pairs:
                    first, second = correct(m, a, q, s), correct(m, b, q, s)
                    ids = sorted(first)
                    result = mcnemar([first[i] for i in ids], [second[i] for i in ids])
                    low, high = diff_ci(first, second, fams[s])
                    delta = (sum(first.values()) - sum(second.values())) / len(ids)
                    unit = " (items)" if fams[s] is None else ""
                    lines.append(f"| {short} | {q} | `{s}` | {a} | {b} | {reads} | {100 * delta:+.1f} | "
                                 f"{result.only_first} / {result.only_second} | {result.p_exact:.3g} | "
                                 f"[{100 * low:+.1f}, {100 * high:+.1f}]{unit} |")

    # --- ranking ---------------------------------------------------------------------
    lines += ["", f"#### Qwen against Llama, every variant, at {DEPLOYED_QUANT}", "",
              "| Variant | Split | EM Qwen | EM Llama | only Qwen / only Llama | exact p | 95% CI Qwen - Llama (pp) |",
              "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |"]
    qwen, llama = list(MODELS)
    for v in VARIANTS:
        for s in SPLITS:
            first, second = correct(qwen, v, DEPLOYED_QUANT, s), correct(llama, v, DEPLOYED_QUANT, s)
            ids = sorted(first)
            result = mcnemar([first[i] for i in ids], [second[i] for i in ids])
            low, high = diff_ci(first, second, fams[s])
            lines.append(f"| {v} | `{s}` | {sum(first.values()) / len(ids):.3f} | "
                         f"{sum(second.values()) / len(ids):.3f} | {result.only_first} / "
                         f"{result.only_second} | {result.p_exact:.3g} | "
                         f"[{100 * low:+.1f}, {100 * high:+.1f}]{' (items)' if fams[s] is None else ''} |")

    # --- rotation by direction ---------------------------------------------------------
    for s in IN_DOMAIN:
        rot = [r for r in split_rows(s) if r["gold_intent"] == "rotate"]
        by_cue: dict[str, list[str]] = defaultdict(list)
        for r in rot:
            by_cue[cue(r["transcript"])].append(r["id"])
        cols = [(m, v) for m in MODELS for v in VARIANTS]
        for q in QUANTS:
            lines += ["", f"#### `{s}` rotate items by direction cue, errors at {q}", "",
                      "| Cue | items | " + " | ".join(f"{MODELS[m]} {v}" for m, v in cols) + " |",
                      "| :--- | ---: | " + " | ".join("---:" for _ in cols) + " |"]
            for name in (*(n for n, _ in CUES), "none"):
                ids = by_cue.get(name)
                if not ids:
                    continue
                lines.append(f"| {name} | {len(ids)} | " + " | ".join(
                    str(sum(not data[(m, v, q, s)][i]["correct"] for i in ids)) for m, v in cols) + " |")
            lines.append(f"| all | {len(rot)} | " + " | ".join(
                str(sum(not data[(m, v, q, s)][r['id']]['correct'] for r in rot)) for m, v in cols) + " |")

    # --- yaw sign errors across the in-domain sets ----------------------------------------
    lines += ["", "#### Errors and yaw sign errors on the two in-domain sets", "",
              "Sign error: the prediction equals the gold command except that the yaw has the",
              "opposite sign. Pooled over `test_golden` and `test_synth` (440 items).", "",
              "| Model | Quant | Variant | errors | yaw sign errors |",
              "| :--- | :--- | :--- | ---: | ---: |"]
    for m, short in MODELS.items():
        for q in QUANTS:
            for v in VARIANTS:
                recs = [r for s in IN_DOMAIN for r in data[(m, v, q, s)].values()]
                lines.append(f"| {short} | {q} | {v} | {sum(not r['correct'] for r in recs)} | "
                             f"{sum(is_sign_error(r) for r in recs)} |")
    return "\n".join(lines) + "\n"


def run() -> int:
    text = markdown()
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
