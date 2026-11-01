#!/usr/bin/env python3
"""Shared scoring and tables for the A-M2 baselines (`eval/baseline_rules.py`,
`eval/baseline_fewshot.py`).

A baseline is scored exactly as the fine-tuned artefacts are, so the comparison is
like for like:

- **Text splits.** Canonical exact match through `eval.metrics` (as `eval/surface_b.py`),
  on `test_golden` (200), `test_synth` (240) and `test_ood` (150). The fine-tuned rows
  are re-scored here from `results/surface_b_preds/`, grammar on, so both sides pass
  through the same code in the same run.
- **Intervals.** Family-clustered percentile bootstrap (`eval.stats.cluster_bootstrap_ci`)
  over the 12 template families of the golden set (`eval.error_families.families()`)
  and, likewise, over the 12 families of the synthetic set. The out-of-domain set is one
  family, so its interval is item-level and labelled so.
- **Paired tests.** Exact McNemar against the deployed artefact on the same items,
  item-level.
- **Speech-recognised transcripts.** Exp-3's per-segment transcripts from the Pi run
  (`results/exp3_pi_preds/`), scored by `eval.exp3.score_segment` and `outcome`, as
  `eval/asr_all_artefacts.py` does for the six artefacts, whose per-segment output this
  module reads for the comparison rows.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

from eval.error_families import families
from eval.exp3 import outcome, score_segment
from eval.metrics import (Prediction, false_command_rate, intent_macro_f1, safe_failure_rate,
                          score_predictions, slot_micro_f1)
from eval.stats import bootstrap_ratio_ci, cluster_bootstrap_ci, mcnemar

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
SURFACE_B = REPO / "results" / "surface_b_preds"
ASR_SOURCE = REPO / "results" / "exp3_pi_preds"
ASR_ARTEFACTS = REPO / "results" / "asr_all_artefacts_preds"

SPLITS = ("test_golden", "test_synth", "test_ood")
ASR_CONDITIONS = ("clean", "20", "15", "10", "5")
DEPLOYED = "qwen2.5-0.5b-instruct-Q4_K_M"
#: The three fine-tuned models at the deployed quantisation, for side-by-side rows.
FINE_TUNED = ("qwen2.5-0.5b-instruct-Q4_K_M", "llama-3.2-1b-instruct-Q4_K_M",
              "smollm2-360m-instruct-Q4_K_M")
SHORT = {"qwen2.5-0.5b-instruct-Q4_K_M": "Qwen2.5-0.5B FT Q4_K_M",
         "llama-3.2-1b-instruct-Q4_K_M": "Llama-3.2-1B FT Q4_K_M",
         "smollm2-360m-instruct-Q4_K_M": "SmolLM2-360M FT Q4_K_M"}


def quiet() -> None:
    """`schema.validate` logs every clamp and fallback; re-scoring buries the output."""
    logging.getLogger("schema.validate").setLevel(logging.ERROR)
    logging.disable(logging.WARNING)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_items(split: str) -> list[dict]:
    return read_jsonl(DATA / f"{split}.jsonl")


def split_families(split: str) -> dict[str, list[str]] | None:
    """Clusters for the interval: None for the single-family out-of-domain set."""
    if split == "test_golden":
        return families()
    fams: dict[str, list[str]] = defaultdict(list)
    for row in split_items(split):
        fams[row["template_family"]].append(row["id"])
    return dict(fams) if len(fams) > 1 else None


# --- text splits -------------------------------------------------------------


def predictions_from_rows(rows: Sequence[dict]) -> dict[str, Prediction]:
    """`rows` carry `id`, `gold` and `raw`, as `results/surface_b_preds/` does."""
    scored = score_predictions([r["raw"] for r in rows], [r["gold"] for r in rows])
    return {r["id"]: p for r, p in zip(rows, scored)}


def fine_tuned_predictions(stem: str, split: str) -> dict[str, Prediction]:
    return predictions_from_rows(read_jsonl(SURFACE_B / f"{stem}_{split}.jsonl"))


def em_interval(correct: Mapping[str, bool], split: str) -> tuple[float, float, str]:
    fams = split_families(split)
    if fams is None:
        hits = [float(v) for v in correct.values()]
        return (*bootstrap_ratio_ci(hits, [1.0] * len(hits)), "items")
    lo, hi = cluster_bootstrap_ci([(float(sum(correct[i] for i in ids)), float(len(ids)))
                                   for ids in fams.values()])
    return lo, hi, "families"


def difference_ci(first: Mapping[str, bool], second: Mapping[str, bool],
                  fams: Mapping[str, Sequence[str]]) -> tuple[float, float, float]:
    """`first - second` over the items of `fams`, with the families resampled whole."""
    clusters = [(float(sum(first[i] - second[i] for i in ids)), float(len(ids)))
                for ids in fams.values()]
    total = sum(n for _, n in clusters)
    return (sum(d for d, _ in clusters) / total, *cluster_bootstrap_ci(clusters))


def failure_kind(p: Prediction) -> str:
    if p.correct:
        return "correct"
    if p.predicted is None:
        return "not a command"
    predicted = json.loads(p.predicted)["intent"]
    if predicted != p.gold_intent:
        return "wrong intent"
    gold = json.loads(p.gold)
    decoded = json.loads(p.predicted)
    if {k: v for k, v in decoded.items() if k != "ids"} == {k: v for k, v in gold.items() if k != "ids"}:
        return "only the ids wrong"
    return "right intent, other slot wrong"


def text_tables(systems: Mapping[str, Mapping[str, Mapping[str, Prediction]]],
                reference: str) -> list[str]:
    """`systems[name][split][id]`. `reference` names the system McNemar pairs against."""
    lines = ["#### Reference transcripts", "",
             "EM: canonical exact match. CI: 95% percentile bootstrap, over the 12 template "
             "families (golden, synthetic) or over items (out-of-domain, one family). "
             "Safe: failed items that dispatched only `unknown` or `hover`. False cmd: "
             "out-of-domain items that dispatched anything but `unknown` (NFR-18).", "",
             "| System | Split | n | EM | 95% CI | Intent F1 | Slot F1 | Safe failures "
             "| False cmd |",
             "| :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |"]
    for name, by_split in systems.items():
        for split in SPLITS:
            preds = by_split[split]
            correct = {i: p.correct for i, p in preds.items()}
            lo, hi, unit = em_interval(correct, split)
            values = list(preds.values())
            failed = [p for p in values if not p.correct]
            safe = sum(p.dispatched in ("unknown", "hover") for p in failed)
            fcr = f"{false_command_rate(values):.3f}" if split == "test_ood" else ""
            lines.append(
                f"| {name} | `{split}` | {len(values)} | {sum(correct.values()) / len(values):.3f} "
                f"| [{lo:.3f}, {hi:.3f}] ({unit}) | {intent_macro_f1(values):.3f} "
                f"| {slot_micro_f1(values):.3f} | {safe}/{len(failed)} | {fcr} |")

    lines += ["", f"#### Paired against {reference}, exact McNemar (item-level)", "",
              f"b: the row's system right and {reference} wrong; c: the reverse. "
              "Item-level p; items of one family are not independent, so read it with "
              "the clustered intervals above.", "",
              "Difference: the row's EM minus the reference's, in points, with a 95% interval "
              "that resamples the families whole (items on the out-of-domain set).", "",
              "| System | Split | b | c | p (exact, item-level) | Difference | 95% CI |",
              "| :--- | :--- | ---: | ---: | ---: | ---: | :--- |"]
    for name, by_split in systems.items():
        if name == reference:
            continue
        for split in SPLITS:
            ids = list(systems[reference][split])
            first = {i: by_split[split][i].correct for i in ids}
            second = {i: systems[reference][split][i].correct for i in ids}
            result = mcnemar([first[i] for i in ids], [second[i] for i in ids])
            fams = split_families(split) or {i: [i] for i in ids}
            diff, lo, hi = difference_ci(first, second, fams)
            lines.append(f"| {name} | `{split}` | {result.only_first} | {result.only_second} "
                         f"| {result.p_exact:.3g} | {100 * diff:+.1f} "
                         f"| [{100 * lo:+.1f}, {100 * hi:+.1f}] |")

    kinds = ("wrong intent", "right intent, other slot wrong", "only the ids wrong", "not a command")
    lines += ["", "#### How the failures fail", "",
              "| System | Split | failed | " + " | ".join(kinds) + " |",
              "| :--- | :--- | ---: | " + " | ".join("---:" for _ in kinds) + " |"]
    for name, by_split in systems.items():
        for split in SPLITS:
            counts = Counter(failure_kind(p) for p in by_split[split].values())
            failed = sum(v for k, v in counts.items() if k != "correct")
            lines.append(f"| {name} | `{split}` | {failed} | "
                         + " | ".join(str(counts[k]) for k in kinds) + " |")

    ood = split_items("test_ood")
    sources = sorted({row["variant"] for row in ood})
    lines += ["", "#### Out-of-domain set, false commands per source", "",
              "| System | " + " | ".join(f"{s} (n = {sum(r['variant'] == s for r in ood)})"
                                         for s in sources) + " |",
              "| :--- | " + " | ".join("---:" for _ in sources) + " |"]
    for name, by_split in systems.items():
        cells = [str(sum(by_split["test_ood"][r["id"]].dispatched != "unknown"
                         for r in ood if r["variant"] == s)) for s in sources]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    golden = {row["id"]: row for row in split_items("test_golden")}
    lines += ["", "#### Golden set, correct items per template family", "",
              "| Family | Intent | n | " + " | ".join(systems) + " |",
              "| :--- | :--- | ---: | " + " | ".join("---:" for _ in systems) + " |"]
    for fam, ids in families().items():
        intent = Counter(golden[i]["gold_intent"] for i in ids).most_common(1)[0][0]
        cells = [str(sum(systems[n]["test_golden"][i].correct for i in ids)) for n in systems]
        lines.append(f"| {fam} | {intent} | {len(ids)} | " + " | ".join(cells) + " |")

    by_intent = defaultdict(list)
    for split in ("test_golden", "test_synth"):
        for row in split_items(split):
            by_intent[row["gold_intent"]].append((split, row["id"]))
    lines += ["", "#### Golden and synthetic sets, exact match per gold intent", "",
              "| Intent | n | " + " | ".join(systems) + " |",
              "| :--- | ---: | " + " | ".join("---:" for _ in systems) + " |"]
    for intent in sorted(by_intent):
        keys = by_intent[intent]
        cells = [f"{sum(systems[n][s][i].correct for s, i in keys) / len(keys):.2f}"
                 for n in systems]
        lines.append(f"| {intent} | {len(keys)} | " + " | ".join(cells) + " |")
    return lines


# --- speech-recognised transcripts ------------------------------------------


def run_asr(parse: Callable[[str], str], out_dir: Path) -> None:
    """Parse every recorded segment transcript with `parse` and score it as Exp-3 does."""
    for condition in ("text", *ASR_CONDITIONS):
        rows = []
        for source in read_jsonl(ASR_SOURCE / f"{condition}.jsonl"):
            segments = []
            for seg in source["segments"]:
                raw = parse(seg["transcript"])
                segments.append({"transcript": seg["transcript"], "raw": raw,
                                 **score_segment(raw, source["gold"])})
            rows.append({"id": source["id"], "condition": condition, "gold": source["gold"],
                         "segments": segments})
        write_jsonl(out_dir / f"{condition}.jsonl", rows)


def asr_outcomes(path: Path) -> dict[str, object]:
    return {r["id"]: outcome(r["segments"]) for r in read_jsonl(path)}


def asr_tables(systems: Mapping[str, Path], reference: str) -> list[str]:
    """`systems[name]` is a directory of `<condition>.jsonl` in Exp-3's row format."""
    fams = families()
    lines = ["#### Speech-recognised transcripts (Exp-3 Pi run)", "",
             "Whisper `tiny.en` transcripts recorded on the Pi (`results/exp3_pi_preds/`), "
             "one per endpointed segment; each system parses the same segments. Scoring is "
             "Exp-3's rule of record (`eval.exp3.outcome`): an item is recognised only if "
             "it produced a segment and every dispatched command was correct. n = 200 "
             "golden items; CI over the 12 families. EM is the same system on the "
             "reference transcripts of that run.", "",
             "| System | SNR | EM | CRR | 95% CI | EM - CRR | Safe failures |",
             "| :--- | :--- | ---: | ---: | :--- | ---: | :--- |"]
    crr_by: dict[str, dict[str, dict[str, bool]]] = defaultdict(dict)
    for name, directory in systems.items():
        em = {i: o.correct for i, o in asr_outcomes(directory / "text.jsonl").items()}
        for condition in ASR_CONDITIONS:
            outs = asr_outcomes(directory / f"{condition}.jsonl")
            crr = {i: o.correct for i, o in outs.items()}
            crr_by[condition][name] = crr
            failed = [o for o in outs.values() if not o.correct]
            lo, hi = cluster_bootstrap_ci([(float(sum(crr[i] for i in ids)), float(len(ids)))
                                           for ids in fams.values()])
            e, c = sum(em.values()) / len(em), sum(crr.values()) / len(crr)
            label = "clean" if condition == "clean" else f"{condition} dB"
            lines.append(f"| {name} | {label} | {e:.3f} | {c:.3f} | [{lo:.3f}, {hi:.3f}] "
                         f"| {e - c:+.3f} | {sum(o.safe for o in failed)}/{len(failed)} |")

    lines += ["", f"Paired against {reference} at each SNR, exact McNemar (item-level):", "",
              "| System | SNR | b | c | p (exact, item-level) | Difference | 95% CI, families |",
              "| :--- | :--- | ---: | ---: | ---: | ---: | :--- |"]
    for name in systems:
        if name == reference:
            continue
        for condition in ASR_CONDITIONS:
            ref = crr_by[condition][reference]
            ids = list(ref)
            result = mcnemar([crr_by[condition][name][i] for i in ids], [ref[i] for i in ids])
            label = "clean" if condition == "clean" else f"{condition} dB"
            diff, lo, hi = difference_ci(crr_by[condition][name], ref, fams)
            lines.append(f"| {name} | {label} | {result.only_first} | {result.only_second} "
                         f"| {result.p_exact:.3g} | {100 * diff:+.1f} "
                         f"| [{100 * lo:+.1f}, {100 * hi:+.1f}] |")

    lines += ["", "Clean audio, items recognised per template family:", "",
              "| Family | n | " + " | ".join(systems) + " |",
              "| :--- | ---: | " + " | ".join("---:" for _ in systems) + " |"]
    for fam, ids in fams.items():
        cells = [str(sum(crr_by["clean"][n][i] for i in ids)) for n in systems]
        lines.append(f"| {fam} | {len(ids)} | " + " | ".join(cells) + " |")
    return lines
