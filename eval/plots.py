#!/usr/bin/env python3
"""Every figure, regenerated from `results/`. No figure is ever edited by hand.

**Figure 2 --- the accuracy/latency Pareto plot** (prd.md 8.2): exact match against
p95 latency, marker area proportional to peak RSS, one point per model x
quantisation, with the 2.5 s target as a vertical line.

Three things about that specification do not survive contact with what was actually
measured, and the figure states each of them on its own face rather than in a caption
a reader may not reach:

1. **Three of the six points do not exist.** The Sat-19 triage scoped Exp-1 to Q4_K_M
   (STATE.md), so no Q8_0 artefact has a latency or an RSS figure from the Pi. Their
   accuracy is known and their cost is not, which is not the same as their cost being
   zero --- they are named in a box on the axes instead of being plotted somewhere
   convenient.

2. **The x-axis is not the quantity NFR-2 bounds.** NFR-2's 2,500 ms runs end-to-end
   from end-of-speech and therefore includes VAD, STT, validation, the FSM and the
   bus; Exp-1 measured the SLM stages only (Table 6's prefill and decode rows,
   summed). Every plotted x is therefore a *lower bound* on the quantity the vertical
   line bounds, and a point left of the line has not met NFR-2 --- it has failed to
   rule itself out. Exp-2 measures the real thing. The directly comparable budget,
   drawn as a second line, is Table 6's own SLM allowance of 250 + 1,100 ms.

3. **No trial was thermally throttled, and the figure says so.** All 180 scored
   trials of the cooled run held the 2.4 GHz pinned clock at 67.5-74.1 C, so NFR-10
   passes at 0% against a 5% budget. The predecessor run throttled on 180/180 at a
   1.5 GHz cap and cost 25-41% of decode p95 (`results/thermal_headroom.md`). The
   thermal state stays on the figure for the same reason it did when it was bad: a
   Pareto plot invites the reader to treat the x-axis as a property of the model,
   and here it is partly a property of the enclosure.

**The keyword-spotter operating curve** (Ingenieur Ch 5, `fig:keyword-curve`): for each
reflex-path class, the false-reject rate on the synthetic test takes against the false
accepts per hour on the ambient-speech stream, as the common threshold sweeps the
selection grid, with the operating threshold marked. Everything plotted is read from
`results/wake_training.json`, which `train/train_wake.py` writes; the threshold is the
one that script selected on val, never re-selected here. Two things the figure states
on its face:

1. **The operating point's false-accept rate carries its interval.** One event in 0.43 h
   gives a 95% interval that straddles the 1/h budget, and that is the whole reason the
   criterion is "not demonstrated". A bare marker would read as a miss.

2. **The top of the curve is flat in false accepts.** The one test false accept survives
   every threshold from 0.88 up, so the points pile up at one x; they are drawn, not
   thinned, because that pile is the finding.

**The Master journal-article figures** (`--article`): the Pareto plot at one-column width,
exact match by precision with the parameter-matched control, the prefill/decode split,
cooled against uncooled timing, and out-of-domain abstention, written to
`articles/master/figures/`, and the Ingenieur article's keyword curve, two-path latency,
recognition against noise and formation box plots, written to `articles/ingenieur/figures/`.

Usage:
    python eval/plots.py
    python eval/plots.py --article
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # no display on the workstation or in CI
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

#: TrueType, not Type 3, in the PDF: Type 3 fonts are vector but render poorly in some
#: viewers and are rejected by some submission checkers.
matplotlib.rcParams["pdf.fonttype"] = 42

if __package__ in (None, ""):   # `python eval/plots.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.bench import aggregate, read_trials
from eval.stats import nearest_rank

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SURFACE_B = RESULTS / "surface_b.csv"
#: The cooled run of record -- the PRD declares a Pi 5 with an active cooler
#: (PRD.tex:199) and names one as the mitigation for risk R-8 (PRD.tex:1841). The first
#: Exp-1 predated the cooler and throttled on 180/180 trials; it is kept and reported in
#: `results/thermal_headroom.md`, not plotted, because a Pareto frontier drawn from it
#: describes hardware this project never declared.
EXP1 = RESULTS / "exp1_cooled.csv"
FIGURE2 = RESULTS / "figure2_pareto"

#: The split Figure 2 plots. `test_golden` is real recorded speech; `test_synth`
#: shares template families with training and would flatter every point equally.
SPLIT = "test_golden"

#: NFR-2, end-to-end from end-of-speech. Drawn as specified, and drawn dashed and
#: annotated because the plotted x does not yet include the non-SLM stages.
NFR2_MS = 2500.0

#: Table 6's SLM allowance: prefill p95 250 ms + decode p95 1,100 ms. This one *is*
#: measured on the same quantity as the x-axis, which is why it is the solid line.
SLM_BUDGET_MS = 250.0 + 1100.0

#: NFR-4. The accuracy constraint of record; a configuration below it is not
#: deployable however fast it is, which is what makes this a Pareto plot and not a
#: latency ranking.
NFR4_EM = 0.85

#: The names the thesis prose uses. Each point is labelled, so one marker shape serves
#: every model and the legend needs to explain only fill (frontier) and area (memory).
#: The keyword-spotter record: held-out rates at every grid threshold, and the
#: operating threshold `train/train_wake.py` selected on val before test was scored.
WAKE_TRAINING = RESULTS / "wake_training.json"
FIGURE_KEYWORD = RESULTS / "figure_keyword_curve"

#: The two keyword criteria of the Ingenieur's tab:nonfunctional-requirements: false
#: accepts on ambient speech (both classes together) and false rejects per class.
KEYWORD_FA_BUDGET_PER_HOUR = 1.0
KEYWORD_FRR_BUDGET = 0.10

#: Class names as the thesis prints them, in the order the legend lists them.
KEYWORD_CLASSES = {"swarm_hold": "swarm hold", "swarm_abort": "swarm abort"}

DISPLAY = {"llama-3.2-1b-instruct": "Llama-3.2-1B", "qwen2.5-0.5b-instruct": "Qwen2.5-0.5B",
           "smollm2-360m-instruct": "SmolLM2-360M"}


def _display(path: Path) -> str:
    """Repo-relative when it can be, absolute when it cannot.

    `Path.relative_to` raises on a path outside the repository, which would replace
    an error message naming the missing sweep with an unrelated `ValueError` about
    subpaths --- the failure mode hiding the failure it was written to report.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing {_display(path)} -- run the sweep that writes it")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _exp1_summaries() -> dict[str, dict]:
    """Config stem -> Exp-1 summary, warm-up rows excluded by name, as Table 17 does."""
    rows = [r for r in read_trials(EXP1) if not r["config"].endswith("-warmup")]
    return {s["config"]: s for s in aggregate(rows)}


def _points() -> tuple[list[dict], list[dict]]:
    """(plottable, unmeasured) --- one entry per model x quantisation on `SPLIT`.

    A configuration is plottable when Exp-1 produced a latency for it. The split is
    deliberate: an unmeasured configuration is carried through with its accuracy so
    the figure can say it exists, rather than being dropped and silently reducing a
    six-point specification to a three-point plot.
    """
    exp1 = _exp1_summaries()
    plottable, unmeasured = [], []
    for row in _rows(SURFACE_B):
        if row["grammar"] != "on" or row["split"] != SPLIT:
            continue
        entry = {"model": row["model"], "quant": row["quant"],
                 "em": float(row["exact_match"]), "n_items": row["n"]}
        summary = exp1.get(f"{row['model']}-{row['quant']}")
        if summary and summary.get("total_p95") not in ("", None):
            entry["p95_ms"] = float(summary["total_p95"])
            entry["rss_mb"] = float(summary["peak_rss_mb"])
            entry["throttled"] = float(summary["throttled_fraction"])
            entry["n_latency"] = int(summary["n_latency"])
            plottable.append(entry)
        else:
            unmeasured.append(entry)
    return plottable, unmeasured


def pareto_frontier(points: list[dict]) -> list[dict]:
    """Non-dominated points: no other point is both faster and at least as accurate.

    Ties are kept rather than broken --- two configurations with the same accuracy and
    the same latency are both on the frontier, and picking one is the selection rule's
    job (prd.md 10.2), not the plot's.
    """
    frontier = [
        point for point in points
        if not any(other is not point
                   and other["p95_ms"] <= point["p95_ms"]
                   and other["em"] >= point["em"]
                   and (other["p95_ms"] < point["p95_ms"] or other["em"] > point["em"])
                   for other in points)
    ]
    return sorted(frontier, key=lambda p: p["p95_ms"])


def figure2() -> list[Path]:
    plottable, unmeasured = _points()
    if not plottable:
        raise SystemExit("no configuration has both an accuracy and an Exp-1 latency")

    figure, axes = plt.subplots(figsize=(7.2, 5.0))

    #: Marker *area* scales with RSS so the visual quantity a reader compares is the
    #: one the legend claims. Scaling the radius instead would make a 2x memory
    #: footprint look like a 4x one. The constant is chosen so the largest artefact
    #: measured (1.63 GB) is a legible dot rather than a blob covering its neighbours.
    area = lambda rss: 0.22 * rss

    frontier = pareto_frontier(plottable)
    if len(frontier) > 1:
        axes.plot([p["p95_ms"] for p in frontier], [p["em"] for p in frontier],
                  color="0.45", linewidth=1.0, linestyle="-", zorder=1,
                  label="Pareto frontier")

    for point in plottable:
        dominated = point not in frontier
        axes.scatter(point["p95_ms"], point["em"], s=area(point["rss_mb"]),
                     marker="o",
                     facecolor="white" if dominated else "#1f6feb",
                     edgecolor="#1f6feb", linewidth=1.6, zorder=3,
                     alpha=0.95)
        #: Label to the left of the marker, vertically centred: a label centred above or
        #: below spans about 700 ms of the axis and crossed a budget line on two of the
        #: three points. Every point sits right of 800 ms, so the left side has room.
        pad = 6 + (area(point["rss_mb"]) ** 0.5) / 2
        axes.annotate(
            f"{DISPLAY.get(point['model'], point['model'])} {point['quant']}\n"
            f"EM {point['em']:.3f} · p95 {point['p95_ms']:,.0f} ms\n"
            f"{point['rss_mb'] / 1024:.2f} GiB",
            xy=(point["p95_ms"], point["em"]),
            xytext=(-pad, 0), textcoords="offset points", ha="right", va="center",
            fontsize=7.5, color="0.15")

    axes.set_ylim(0.55, 1.0)
    axes.set_xlim(0, max(NFR2_MS, max(p["p95_ms"] for p in plottable)) * 1.12)

    #: Line labels sit in the top margin, above the plotting area, so that no label can
    #: land on a point annotation (the rotated in-axes labels did, on two points).
    top = blended_transform_factory(axes.transData, axes.transAxes)
    axes.axvline(SLM_BUDGET_MS, color="#b45309", linewidth=1.2, zorder=2)
    axes.text(SLM_BUDGET_MS, 1.015, "Language-model stage allowance\n1,350 ms (same quantity as the axis)",
              transform=top, ha="center", va="bottom", fontsize=7, color="#b45309")

    axes.axvline(NFR2_MS, color="#b91c1c", linewidth=1.4, linestyle="--", zorder=2)
    axes.text(NFR2_MS, 1.015, "End-to-end budget 2,500 ms\n(the axis measures part of it)",
              transform=top, ha="center", va="bottom", fontsize=7, color="#b91c1c")

    axes.axhline(NFR4_EM, color="#15803d", linewidth=1.2, linestyle=":", zorder=2)
    axes.text(40, NFR4_EM + 0.004, "Exact-match threshold 0.85", ha="left",
              va="bottom", fontsize=7.5, color="#15803d")

    if unmeasured:
        names = "\n".join(f"  · {DISPLAY.get(p['model'], p['model'])} {p['quant']}"
                          f"  EM {p['em']:.3f}" for p in unmeasured)
        axes.text(0.015, 0.03,
                  "Not plotted, not timed on the Raspberry Pi 5:\n" + names,
                  transform=axes.transAxes, fontsize=7.5, va="bottom", ha="left",
                  color="0.25",
                  bbox=dict(boxstyle="round,pad=0.45", facecolor="#f6f6f6",
                            edgecolor="0.75", linewidth=0.7))

    #: No in-figure title: the thesis caption names and numbers the figure, and a title
    #: typed here ("Figure 2") cannot follow the document's own numbering. The thermal
    #: state the title used to carry is in the caption.
    n_latency = {p["n_latency"] for p in plottable}
    n_items = {int(p["n_items"]) for p in plottable}
    axes.set_xlabel("Language-model prefill + decode, p95 (ms)\n"
                    f"Raspberry Pi 5, n = {min(n_latency)} timed runs per configuration")
    axes.set_ylabel(f"Exact match on {SPLIT} (n = {min(n_items)} items)")
    axes.grid(True, linewidth=0.4, color="0.9", zorder=0)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)

    #: A size legend, not a colour one: marker area is the only channel carrying RSS,
    #: and a reader cannot decode it from the points alone.
    handles = [axes.scatter([], [], s=area(mb), facecolor="white", edgecolor="0.4",
                            linewidth=1.2, label=f"{mb / 1024:.1f} GiB peak memory")
               for mb in (512, 1024, 1536)]
    if len(frontier) > 1:
        handles.insert(0, axes.plot([], [], color="0.45", linewidth=1.0,
                                    label="Pareto frontier")[0])
    handles += [
        axes.scatter([], [], s=70, facecolor="#1f6feb", edgecolor="#1f6feb",
                     linewidth=1.6, label="non-dominated"),
        axes.scatter([], [], s=70, facecolor="white", edgecolor="#1f6feb",
                     linewidth=1.6, label="dominated"),
    ]
    #: Anchored in the empty band between the two budget lines, below the threshold.
    axes.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.52, 0.02), fontsize=7, frameon=True,
                borderpad=0.6, labelspacing=0.9, handletextpad=1.0, framealpha=0.95)

    figure.tight_layout()
    written = []
    for suffix in (".pdf", ".png"):
        path = FIGURE2.with_suffix(suffix)
        figure.savefig(path, dpi=200)
        written.append(path)
    plt.close(figure)
    return written


def keyword_curve(split: str = "test") -> dict:
    """The operating curve of one split, sorted by threshold, plus what the figure marks.

    Returned rather than plotted so the tests can check the data the figure is drawn
    from: which point is the operating point, and that the interval comes from the file.
    """
    if not WAKE_TRAINING.is_file():
        raise SystemExit(f"missing {_display(WAKE_TRAINING)} -- run train/train_wake.py")
    record = json.loads(WAKE_TRAINING.read_text(encoding="utf-8"))
    points = sorted(record["roc"][split], key=lambda p: p["threshold"])
    threshold = record["threshold"]
    operating = [p for p in points if p["threshold"] == threshold]
    if len(operating) != 1:
        raise SystemExit(f"operating threshold {threshold} is not on the {split} grid")
    return {"points": points, "threshold": threshold, "operating": operating[0],
            "fa_ci95": record[split]["ambient"]["per_hour_ci95"],
            "hours": record[split]["ambient"]["hours"],
            "n_positives": record[split]["positives"]["n"]}


def figure_keyword_curve() -> list[Path]:
    curve = keyword_curve("test")
    points, operating = curve["points"], curve["operating"]
    colours = {"swarm_hold": "#1f6feb", "swarm_abort": "#b45309"}

    figure, axes = plt.subplots(figsize=(7.2, 4.6))

    #: The interval first, so the curves draw over it. Shaded across the full height:
    #: it is an interval on the x-axis only, and a bar at one class's y would suggest
    #: it belonged to that class.
    low, high = curve["fa_ci95"]
    axes.axvspan(low, high, color="0.5", alpha=0.12, linewidth=0, zorder=0)
    #: Label inside the band, left of the budget line, so the line does not cross it.
    axes.text(low * 1.25, 0.045,
              "95% interval of the\nfalse-accept rate at the\noperating threshold",
              ha="left", va="top", fontsize=7, color="0.3")

    for key, label in KEYWORD_CLASSES.items():
        xs = [p["fa_per_hour_ambient"] for p in points]
        ys = [p["frr"][key] for p in points]
        n = curve["n_positives"][key]
        axes.plot(xs, ys, color=colours[key], linewidth=1.2, marker="o", markersize=2.5,
                  zorder=3, label=f"{label} (n = {n} test takes)")
        axes.scatter(operating["fa_per_hour_ambient"], operating["frr"][key], s=70,
                     facecolor="white", edgecolor=colours[key], linewidth=1.8, zorder=4)

    #: Two thresholds named on the curve, both read from the grid: the lowest, where
    #: the curve starts, and the operating one. Naming more would crowd the flat top.
    first = points[0]
    axes.annotate(f"threshold {first['threshold']:g}",
                  xy=(first["fa_per_hour_ambient"], first["frr"]["swarm_hold"]),
                  xytext=(0, 10), textcoords="offset points", ha="center",
                  fontsize=7, color="0.25")
    axes.annotate(f"operating threshold {curve['threshold']:g}\n"
                  f"{operating['fa_per_hour_ambient']:.2f} per hour; false rejects "
                  f"{operating['frr']['swarm_hold']:.3f} (hold), "
                  f"{operating['frr']['swarm_abort']:.3f} (abort)",
                  xy=(operating["fa_per_hour_ambient"], operating["frr"]["swarm_hold"]),
                  xytext=(14, 16), textcoords="offset points", ha="left", fontsize=7,
                  color="0.15", arrowprops=dict(arrowstyle="-", color="0.5", linewidth=0.6))

    axes.axvline(KEYWORD_FA_BUDGET_PER_HOUR, color="#b91c1c", linewidth=1.3, linestyle="--",
                 zorder=2)
    axes.text(KEYWORD_FA_BUDGET_PER_HOUR * 0.93, 0.082,
              f"false-accept budget\n{KEYWORD_FA_BUDGET_PER_HOUR:g} per hour",
              ha="right", va="top", fontsize=7, color="#b91c1c")
    axes.axhline(KEYWORD_FRR_BUDGET, color="#15803d", linewidth=1.2, linestyle=":", zorder=2)
    axes.text(max(p["fa_per_hour_ambient"] for p in points), KEYWORD_FRR_BUDGET + 0.002,
              f"false-reject budget {KEYWORD_FRR_BUDGET:.2f}", ha="right", va="bottom",
              fontsize=7, color="#15803d")

    #: Log x: the rates span 2.3 to 54 per hour, and the interval reaches down to 0.06.
    #: Every test point is above zero, so no point is lost to the log.
    axes.set_xscale("log")
    #: Plain numbers, not powers of ten: a rate of "0.1 per hour" reads without a
    #: conversion step, and the axis spans only three decades.
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes.set_xlim(low * 0.6, max(p["fa_per_hour_ambient"] for p in points) * 1.6)
    axes.set_ylim(-0.004, 0.118)
    axes.set_xlabel(f"False accepts per hour on ambient speech, both classes together "
                    f"({curve['hours']:.3f} h, test split)")
    axes.set_ylabel("False-reject rate (synthetic test takes)")
    axes.grid(True, which="both", linewidth=0.4, color="0.92", zorder=0)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    axes.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.95,
                bbox_to_anchor=(1.0, 0.86))

    figure.tight_layout()
    written = []
    for suffix in (".pdf", ".png"):
        path = FIGURE_KEYWORD.with_suffix(suffix)
        figure.savefig(path, dpi=200)
        written.append(path)
    plt.close(figure)
    return written


# -----------------------------------------------------------------------------
#  Journal-article figures (articles/master). Same sources as the thesis figures,
#  redrawn for one 8.5 cm column of the Elsevier CAS double-column layout: vector
#  PDF only, 7 pt text, no in-figure title (the caption carries it).
# -----------------------------------------------------------------------------

ARTICLE_MASTER = REPO / "articles" / "master" / "figures"
SURFACE_A = REPO / "train" / "kaggle_out" / "surface_a.csv"
ABSTENTION = RESULTS / "nfr9_nfr18_abstention.csv"
#: The uncooled predecessor of EXP1, plotted only against it in the thermal figure.
EXP1_UNCOOLED = RESULTS / "exp1.csv"

#: One column of cas-dc is 8.5 cm.
COLUMN_IN = 8.5 / 2.54

#: Categorical slots 1 and 2 of the validated dataviz palette, and an ordinal blue
#: ramp (light -> dark) for the three precisions, which are ordered, not unordered.
BLUE, ORANGE = "#2a78d6", "#eb6834"
PRECISION_RAMP = {"fp16": "#86b6ef", "Q8_0": "#2a78d6", "Q4_K_M": "#104281"}
#: Budget and threshold lines are drawn in ink, not in a status colour, and are
#: told apart by line style and a text label.
INK = "0.25"

#: Models in parameter order, as the article's tables list them.
ARTICLE_MODELS = ["smollm2-360m-instruct", "qwen2.5-0.5b-instruct",
                  "h2o-danube3-500m-chat", "llama-3.2-1b-instruct"]
SHORT = {"smollm2-360m-instruct": "SmolLM2\n360M", "qwen2.5-0.5b-instruct": "Qwen2.5\n0.5B",
         "h2o-danube3-500m-chat": "Danube3\n500M", "llama-3.2-1b-instruct": "Llama-3.2\n1B"}

ARTICLE_RC = {"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
              "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
              "xtick.major.width": 0.6, "ytick.major.width": 0.6,
              "xtick.major.size": 2.5, "ytick.major.size": 2.5, "hatch.linewidth": 0.6}


def _article_axes(axes) -> None:
    axes.grid(True, axis="y", linewidth=0.4, color="0.9", zorder=0)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)


def _save_article(figure, stem: str, folder: Path = ARTICLE_MASTER) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.pdf"
    #: No creation date in the file, so regenerating an unchanged figure leaves git clean.
    figure.savefig(path, bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    plt.close(figure)
    return path


def _bar_label(axes, x: float, y: float, text: str, dark: bool = False) -> None:
    """Value printed inside the top of a bar, so nothing drawn above it can collide."""
    axes.text(x, y - 0.015, text, ha="center", va="top", fontsize=5.5, rotation=90,
              color="0.1" if dark else "white",
              bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none")
              if dark else None)


def golden_em_by_precision() -> dict[str, dict[str, float]]:
    """model -> {fp16, Q8_0, Q4_K_M} exact match on `test_golden`, where it exists.

    fp16 is the fine-tuned model on the reference surface (`transformers`, no
    grammar); Q8_0 and Q4_K_M are the GGUF artefacts on the deployed surface under
    the grammar. The control has no deployed artefact, so it carries fp16 only.
    """
    em: dict[str, dict[str, float]] = {m: {} for m in ARTICLE_MODELS}
    for row in _rows(SURFACE_A):
        if row["surface"] == "A_fp16_finetuned" and row["split"] == SPLIT:
            em[row["model"]]["fp16"] = float(row["exact_match"])
    for row in _rows(SURFACE_B):
        if row["grammar"] == "on" and row["split"] == SPLIT:
            em[row["model"]][row["quant"]] = float(row["exact_match"])
    return em


def article_em_bars() -> Path:
    em = golden_em_by_precision()
    with plt.rc_context(ARTICLE_RC):
        figure, axes = plt.subplots(figsize=(COLUMN_IN, 2.3))
        width = 0.26
        for index, model in enumerate(ARTICLE_MODELS):
            present = [p for p in PRECISION_RAMP if p in em[model]]
            offsets = [(k - (len(present) - 1) / 2) * width for k in range(len(present))]
            for offset, precision in zip(offsets, present):
                value = em[model][precision]
                control = model == "h2o-danube3-500m-chat"
                axes.bar(index + offset, value, width=width * 0.92,
                         color="white" if control else PRECISION_RAMP[precision],
                         edgecolor=PRECISION_RAMP["Q8_0"] if control else "white",
                         hatch="////" if control else None, linewidth=0.6, zorder=3)
                _bar_label(axes, index + offset, value, f"{value:.3f}",
                           dark=control or precision == "fp16")
        #: The family contrast is fp16 against fp16, Qwen against the control.
        qwen_fp16 = 1 - width
        control_fp16 = 2.0
        top = 1.0
        axes.plot([qwen_fp16, qwen_fp16, control_fp16, control_fp16],
                  [em["qwen2.5-0.5b-instruct"]["fp16"] + 0.02, top, top,
                   em["h2o-danube3-500m-chat"]["fp16"] + 0.02],
                  color=INK, linewidth=0.6)
        axes.text((qwen_fp16 + control_fp16) / 2, top + 0.008,
                  "fp16 vs fp16: 6.5 pp, exact McNemar p = 0.0023",
                  ha="center", va="bottom", fontsize=5.8, color="0.15")
        axes.set_xticks(range(len(ARTICLE_MODELS)))
        axes.set_xticklabels([SHORT[m] for m in ARTICLE_MODELS])
        axes.set_ylim(0, 1.08)
        axes.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes.set_ylabel("Exact match, golden test set (n = 200)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=PRECISION_RAMP[p], label=label)
                   for p, label in (("fp16", "fp16 (reference)"), ("Q8_0", "Q8_0"),
                                    ("Q4_K_M", "Q4_K_M"))]
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", hatch="////",
                                     edgecolor=PRECISION_RAMP["Q8_0"], linewidth=0.6,
                                     label="fp16, family control"))
        axes.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                    ncol=4, frameon=False, handlelength=1.2, columnspacing=0.9)
        _article_axes(axes)
        return _save_article(figure, "fig_em_bars")


def _scored_trials(path: Path) -> dict[str, list[dict]]:
    by_config: dict[str, list[dict]] = {}
    for row in read_trials(path):
        if not row["config"].endswith("-warmup"):
            by_config.setdefault(row["config"], []).append(row)
    return by_config


def _mean(rows: list[dict], column: str) -> float:
    return sum(float(r[column]) for r in rows) / len(rows)


DEPLOYED = [m for m in ARTICLE_MODELS if m != "h2o-danube3-500m-chat"]


def article_latency_stack() -> Path:
    trials = _scored_trials(EXP1)
    summaries = _exp1_summaries()
    with plt.rc_context(ARTICLE_RC):
        figure, axes = plt.subplots(figsize=(COLUMN_IN, 2.1))
        for index, model in enumerate(DEPLOYED):
            config = f"{model}-Q4_K_M"
            rows = trials[config]
            prefill, decode = _mean(rows, "slm_prefill_ms"), _mean(rows, "slm_decode_ms")
            axes.barh(index, prefill, height=0.5, color=ORANGE, zorder=3,
                      label="prefill (mean)" if index == 0 else None)
            #: A 2-unit gap between the stacked segments, so the boundary reads.
            axes.barh(index, decode, left=prefill, height=0.5, color=BLUE, zorder=3,
                      edgecolor="white", linewidth=0.8,
                      label="decode (mean)" if index == 0 else None)
            p95 = float(summaries[config]["total_p95"])
            axes.plot(p95, index, marker="D", markersize=4, color="0.1", zorder=4,
                      linestyle="none", label="total, p95" if index == 0 else None)
            axes.text(p95 + 45, index, f"{p95:,.0f} ms", va="center", fontsize=6,
                      color="0.15")
            axes.text(prefill + decode - 25, index, f"{prefill + decode:,.0f}",
                      va="center", ha="right", fontsize=5.8, color="white", zorder=5)
        axes.axvline(SLM_BUDGET_MS, color=INK, linewidth=0.8, linestyle="--", zorder=2)
        axes.text(SLM_BUDGET_MS + 30, len(DEPLOYED) - 0.45, "stage allowance\n1,350 ms (p95)",
                  fontsize=5.8, color="0.15", va="center")
        axes.set_yticks(range(len(DEPLOYED)))
        axes.set_yticklabels([SHORT[m].replace("\n", " ") + "\nQ4_K_M" for m in DEPLOYED])
        axes.set_xlim(0, 2700)
        axes.set_ylim(-0.6, len(DEPLOYED) - 0.1)
        axes.set_xlabel("Language-model time per command (ms), Raspberry Pi 5, cooled, n = 60")
        axes.grid(True, axis="x", linewidth=0.4, color="0.9", zorder=0)
        axes.set_axisbelow(True)
        for spine in ("top", "right"):
            axes.spines[spine].set_visible(False)
        axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False,
                    handlelength=1.2)
        return _save_article(figure, "fig_latency_stack")


def _paired_panels(panels, series, value, fmt, stem: str, legend_title=None,
                   legend_panel: int = 0) -> Path:
    """Two stacked panels, each a grouped bar per deployed model and one dashed target.

    `panels` holds (column, y label, target, target label, y top); `series` maps a
    series key to (legend label, colour); `value(model, key, column)` reads a number.
    Stacked rather than side by side: two panels across 8.5 cm leave each model group
    too narrow for its tick label. A value label that the target line would strike
    through is drawn inside the top of its bar instead.
    """
    with plt.rc_context(ARTICLE_RC):
        figure, all_axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.3), sharex=True)
        keys = list(series)
        for axes, (column, label, target, target_label, top) in zip(all_axes, panels):
            for index, model in enumerate(DEPLOYED):
                for k, key in enumerate(keys):
                    v = value(model, key, column)
                    x = index + (k - (len(keys) - 1) / 2) * 0.3
                    axes.bar(x, v, width=0.28, color=series[key][1], zorder=3,
                             label=series[key][0] if index == 0 else None)
                    y = v + top * 0.015
                    if y < target < y + top * 0.075:
                        axes.text(x, v - top * 0.015, fmt(column, v), ha="center",
                                  va="top", fontsize=5.5, color="white", zorder=4)
                    else:
                        axes.text(x, y, fmt(column, v), ha="center", va="bottom",
                                  fontsize=5.5, color="0.15")
            axes.axhline(target, color=INK, linewidth=0.8, linestyle="--", zorder=2)
            axes.text(3.05, target + top * 0.015, target_label, fontsize=5.8,
                      color="0.15", ha="right", va="bottom")
            axes.set_ylabel(label)
            axes.set_ylim(0, top)
            axes.set_xlim(-0.5, 3.1)
            _article_axes(axes)
        all_axes[1].set_xticks(range(len(DEPLOYED)))
        all_axes[1].set_xticklabels([SHORT[m].replace("\n", " ") for m in DEPLOYED])
        all_axes[legend_panel].legend(loc="upper right", frameon=False, title=legend_title,
                                      handlelength=1.2, bbox_to_anchor=(1.0, 1.0))
        figure.align_ylabels(all_axes)
        figure.tight_layout(h_pad=0.6)
        return _save_article(figure, stem)


def article_thermal() -> Path:
    runs = {run: {s["config"]: s for s in aggregate(
                [r for rows in _scored_trials(path).values() for r in rows])}
            for run, path in (("cooled", EXP1), ("uncooled", EXP1_UNCOOLED))}
    panels = (("slm_decode_p95", "Decode p95 (ms)", 1100.0, "allowance\n1,100 ms", 2900),
              ("tokens_per_second", "Throughput (tok/s)", 20.0, "floor 20 tok/s", 47))
    series = {"cooled": ("cooled (0% of trials throttled)", BLUE),
              "uncooled": ("uncooled (100% throttled)", ORANGE)}
    return _paired_panels(
        panels, series,
        lambda model, run, column: float(runs[run][f"{model}-Q4_K_M"][column]),
        lambda column, v: f"{v:,.0f}" if column.endswith("p95") else f"{v:.1f}",
        "fig_thermal", legend_panel=1)


def article_ood() -> Path:
    rows = {(r["model"], r["quant"]): r for r in _rows(ABSTENTION) if r["split"] == "test_ood"}
    panels = (("false_command_rate", "False-command rate", 0.05, "target $\\leq$ 0.05", 0.46),
              ("safe_failure_rate", "Safe-failure rate", 0.70, "target $\\geq$ 0.70", 0.82))
    series = {"Q8_0": ("Q8_0", PRECISION_RAMP["Q8_0"]),
              "Q4_K_M": ("Q4_K_M", PRECISION_RAMP["Q4_K_M"])}
    return _paired_panels(
        panels, series,
        lambda model, quant, column: float(rows[(model, quant)][column]),
        lambda column, v: f"{v:.2f}", "fig_ood")


def article_pareto() -> Path:
    plottable, _ = _points()
    frontier = pareto_frontier(plottable)
    with plt.rc_context(ARTICLE_RC):
        figure, axes = plt.subplots(figsize=(COLUMN_IN, 2.5))
        if len(frontier) > 1:
            axes.plot([p["p95_ms"] for p in frontier], [p["em"] for p in frontier],
                      color="0.55", linewidth=0.8, zorder=1)
        for point in plottable:
            dominated = point not in frontier
            axes.scatter(point["p95_ms"], point["em"], s=34, marker="o",
                         facecolor="white" if dominated else BLUE, edgecolor=BLUE,
                         linewidth=1.2, zorder=3)
            name = DISPLAY.get(point["model"], point["model"])
            #: Qwen and Llama are labelled to their left, SmolLM2 below: the frontier
            #: segment runs down-left from Qwen, and the budget lines sit to the right.
            above = point["model"] != "smollm2-360m-instruct"
            axes.annotate(f"{name}\nEM {point['em']:.3f}\n{point['p95_ms']:,.0f} ms, "
                          f"{point['rss_mb'] / 1024:.2f} GiB",
                          xy=(point["p95_ms"], point["em"]),
                          xytext=(-8, 4) if above else (0, -8), textcoords="offset points",
                          ha="right" if above else "center", va="center" if above else "top",
                          fontsize=5.8, color="0.15")
        axes.axvline(SLM_BUDGET_MS, color=INK, linewidth=0.8, linestyle="-", zorder=2)
        axes.axvline(NFR2_MS, color=INK, linewidth=0.8, linestyle="--", zorder=2)
        axes.axhline(NFR4_EM, color=INK, linewidth=0.8, linestyle=":", zorder=2)
        top = blended_transform_factory(axes.transData, axes.transAxes)
        axes.text(SLM_BUDGET_MS, 1.01, "stage allowance\n1,350 ms", transform=top,
                  ha="center", va="bottom", fontsize=5.8, color="0.15")
        axes.text(NFR2_MS, 1.01, "end-to-end budget\n2,500 ms", transform=top,
                  ha="center", va="bottom", fontsize=5.8, color="0.15")
        axes.text(60, NFR4_EM + 0.004, "EM threshold 0.85", ha="left", va="bottom",
                  fontsize=5.8, color="0.15")
        axes.set_xlim(0, 2900)
        axes.set_ylim(0.66, 0.98)
        axes.set_xlabel("Language-model prefill + decode, p95 (ms), Raspberry Pi 5, n = 60")
        axes.set_ylabel("Exact match, golden test set")
        axes.grid(True, linewidth=0.4, color="0.9", zorder=0)
        axes.set_axisbelow(True)
        for spine in ("top", "right"):
            axes.spines[spine].set_visible(False)
        handles = [axes.scatter([], [], s=34, facecolor=BLUE, edgecolor=BLUE,
                                linewidth=1.2, label="non-dominated"),
                   axes.scatter([], [], s=34, facecolor="white", edgecolor=BLUE,
                                linewidth=1.2, label="dominated")]
        axes.legend(handles=handles, loc="lower right", frameon=False)
        return _save_article(figure, "fig_pareto")


def article_master() -> list[Path]:
    return [article_em_bars(), article_latency_stack(), article_thermal(), article_ood(),
            article_pareto()]


# -----------------------------------------------------------------------------
#  Ingenieur journal-article figures (articles/ingenieur). The architecture is TikZ in
#  the manuscript and the trajectory figure is reused from results/ as it is.
# -----------------------------------------------------------------------------

ARTICLE_INGENIEUR = REPO / "articles" / "ingenieur" / "figures"
EXP2_RAW = RESULTS / "exp2_raw"
EXP3_PI = RESULTS / "exp3_pi.csv"
EXP4 = RESULTS / "exp4.csv"
FORMATIONS = ("circle", "line", "wedge")

#: The latency experiment's budgets: parse path from the end of speech, reflex path from
#: the keyword offset (p95 targets, ms).
PARSE_BUDGET_MS = 2500.0
REFLEX_BUDGET_MS = 150.0

#: Recognition targets: CRR in clean audio and at 10 dB.
CRR_CLEAN_TARGET = 0.80
CRR_10DB_TARGET = 0.65


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing {_display(path)} -- run the experiment that writes it")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def article_keyword_curve() -> Path:
    curve = keyword_curve("test")
    points, operating = curve["points"], curve["operating"]
    colours = {"swarm_hold": BLUE, "swarm_abort": ORANGE}
    with plt.rc_context(ARTICLE_RC):
        figure, axes = plt.subplots(figsize=(COLUMN_IN, 2.4))
        low, high = curve["fa_ci95"]
        axes.axvspan(low, high, color="0.5", alpha=0.12, linewidth=0, zorder=0)
        axes.text(low * 1.15, 0.112, "95% CI of the false-accept\nrate at the operating point",
                  ha="left", va="top", fontsize=5.8, color="0.25")
        for key, label in KEYWORD_CLASSES.items():
            axes.plot([p["fa_per_hour_ambient"] for p in points], [p["frr"][key] for p in points],
                      color=colours[key], linewidth=1.0, marker="o", markersize=2, zorder=3,
                      label=f"{label} (n = {curve['n_positives'][key]})")
            axes.scatter(operating["fa_per_hour_ambient"], operating["frr"][key], s=30,
                         facecolor="white", edgecolor=colours[key], linewidth=1.3, zorder=4)
        axes.annotate(f"operating threshold {curve['threshold']:g}",
                      xy=(operating["fa_per_hour_ambient"], operating["frr"]["swarm_hold"]),
                      xytext=(10, 12), textcoords="offset points", ha="left", fontsize=5.8,
                      color="0.15", arrowprops=dict(arrowstyle="-", color="0.5", linewidth=0.5))
        axes.axvline(KEYWORD_FA_BUDGET_PER_HOUR, color=INK, linewidth=0.8, linestyle="--", zorder=2)
        axes.text(KEYWORD_FA_BUDGET_PER_HOUR * 0.9, 0.06, "false-accept\ntarget 1/h", ha="right",
                  va="center", fontsize=5.8, color="0.15")
        axes.axhline(KEYWORD_FRR_BUDGET, color=INK, linewidth=0.8, linestyle=":", zorder=2)
        axes.text(max(p["fa_per_hour_ambient"] for p in points) * 1.4, KEYWORD_FRR_BUDGET + 0.002,
                  f"false-reject target {KEYWORD_FRR_BUDGET:.2f}", ha="right", va="bottom",
                  fontsize=5.8, color="0.15")
        axes.set_xscale("log")
        axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axes.set_xlim(low * 0.6, max(p["fa_per_hour_ambient"] for p in points) * 1.6)
        axes.set_ylim(-0.004, 0.118)
        axes.set_xlabel(f"False accepts per hour, ambient speech ({curve['hours']:.3f} h)")
        axes.set_ylabel("False-reject rate (synthetic test takes)")
        axes.grid(True, which="both", linewidth=0.4, color="0.92", zorder=0)
        axes.set_axisbelow(True)
        for spine in ("top", "right"):
            axes.spines[spine].set_visible(False)
        axes.legend(loc="center right", frameon=False, bbox_to_anchor=(1.0, 0.62))
        return _save_article(figure, "fig_keyword_curve", ARTICLE_INGENIEUR)


#: Parse-path stages in pipeline order, as (label, function of one segment record).
PARSE_STAGES = (
    ("queue", lambda s: s["queue_ms"]),
    ("recognition", lambda s: s["stt_ms"]),
    ("prefill", lambda s: s["prefill_ms"]),
    ("decode", lambda s: s["decode_ms"]),
    ("overhead + validation", lambda s: s["parse_wall_ms"] - s["prefill_ms"]
     - s["decode_ms"] + s["validate_fsm_ms"]),
)
PARSE_COLOURS = ("#c3c2b7", BLUE, ORANGE, "#1baf7a", "#eda100")


def parse_stage_means() -> tuple[list[tuple[str, float]], float, int]:
    """Mean of each parse-path stage per segment, the p95 of the measured total, and n.

    The stages tile the span from the end of speech to the state machine, so their means
    add up to the mean total; the p95 is taken from the recorded totals, never summed.
    """
    segments = [seg for row in _jsonl(EXP2_RAW / "branch_b.jsonl") for seg in row["segments"]]
    means = [(label, sum(f(s) for s in segments) / len(segments)) for label, f in PARSE_STAGES]
    total_p95 = nearest_rank([s["e2e_t0_ms"] for s in segments], 95)
    return means, total_p95, len(segments)


def reflex_means(condition: str) -> tuple[float, float, float, int]:
    """(spotter delay + quantisation mean, Pi compute + publish mean, total p95, n)."""
    from eval.exp2 import algorithmic_ms, first_own_event

    pairs = [(r, first_own_event(r)) for r in _jsonl(EXP2_RAW / f"{condition}.jsonl")]
    pairs = [(r, e) for r, e in pairs if e is not None]
    algorithmic = [algorithmic_ms(r, e) for r, e in pairs]
    system = [e["latency_offset_ms"] - a for (_, e), a in zip(pairs, algorithmic)]
    total_p95 = nearest_rank([e["latency_offset_ms"] for _, e in pairs], 95)
    return (sum(algorithmic) / len(pairs), sum(system) / len(pairs), total_p95, len(pairs))


def article_latency_paths() -> Path:
    means, parse_p95, n_parse = parse_stage_means()
    with plt.rc_context(ARTICLE_RC):
        figure, (top, bottom) = plt.subplots(
            2, 1, figsize=(COLUMN_IN, 3.5), gridspec_kw={"height_ratios": [1, 1.5]})
        left = 0.0
        for (label, value), colour in zip(means, PARSE_COLOURS):
            top.barh(0, value, left=left, height=0.5, color=colour, edgecolor="white",
                     linewidth=0.8, zorder=3, label=label)
            if value > 250:
                top.text(left + value / 2, 0, f"{value:,.0f}", ha="center", va="center",
                         fontsize=5.8, color="white" if colour != "#c3c2b7" else "0.1", zorder=4)
            left += value
        top.plot(parse_p95, 0, marker="D", markersize=4, color="0.1", zorder=5, linestyle="none")
        top.text(parse_p95 + 40, 0.18, f"p95 {parse_p95:,.0f}", fontsize=5.8, color="0.15")
        top.axvline(PARSE_BUDGET_MS, color=INK, linewidth=0.8, linestyle="--", zorder=2)
        top.text(PARSE_BUDGET_MS, 1.0, "budget 2,500 ms (p95)", ha="center", va="bottom",
                 fontsize=5.8, color="0.15", transform=blended_transform_factory(
                     top.transData, top.transAxes))
        top.set_xlim(0, 3500)
        top.set_ylim(-0.45, 0.6)
        top.set_yticks([0])
        top.set_yticklabels([f"parse path\n(n = {n_parse})"])
        top.set_xlabel("ms from end of speech (mean per stage)")
        top.legend(loc="upper center", bbox_to_anchor=(0.45, -0.75), ncol=3, frameon=False,
                   handlelength=1.0, columnspacing=0.8)
        for index, condition in enumerate(("idle", "loaded")):
            algorithmic, system, p95, n = reflex_means(condition)
            bottom.barh(index, algorithmic, height=0.5, color=BLUE, zorder=3,
                        label="spotter delay + frame quantisation" if index == 0 else None)
            bottom.barh(index, system, left=algorithmic, height=0.5, color=ORANGE,
                        edgecolor="white", linewidth=0.6, zorder=3,
                        label="Pi compute + publish" if index == 0 else None)
            bottom.plot(p95, index, marker="D", markersize=4, color="0.1", zorder=5,
                        linestyle="none", label="total, p95" if index == 0 else None)
            bottom.text(p95 + 8, index + 0.18, f"{p95:,.0f}", fontsize=5.8, color="0.15")
            bottom.text(algorithmic + system + 6, index - 0.02,
                        f"mean {algorithmic:,.0f} + {system:,.0f}", fontsize=5.6, va="center",
                        color="0.15")
        bottom.axvline(REFLEX_BUDGET_MS, color=INK, linewidth=0.8, linestyle="--", zorder=2)
        bottom.text(REFLEX_BUDGET_MS, 1.0, "budget 150 ms (p95)", ha="center", va="bottom",
                    fontsize=5.8, color="0.15", transform=blended_transform_factory(
                        bottom.transData, bottom.transAxes))
        bottom.set_yticks([0, 1])
        bottom.set_yticklabels([f"reflex, idle\n(n = {n})", f"reflex, loaded\n(n = {n})"])
        bottom.set_xlim(0, 640)
        bottom.set_ylim(-0.5, 1.6)
        bottom.set_xlabel("ms from keyword offset")
        bottom.legend(loc="upper center", bbox_to_anchor=(0.4, -0.42), ncol=2, frameon=False,
                      handlelength=1.0, columnspacing=0.8)
        for axes, tag in ((top, "(a)"), (bottom, "(b)")):
            axes.grid(True, axis="x", linewidth=0.4, color="0.9", zorder=0)
            axes.set_axisbelow(True)
            for spine in ("top", "right"):
                axes.spines[spine].set_visible(False)
            axes.text(-0.02, 1.02, tag, transform=axes.transAxes, fontsize=7, ha="right",
                      va="bottom", fontweight="bold")
        figure.subplots_adjust(left=0.24, right=0.97, top=0.94, bottom=0.2, hspace=1.35)
        return _save_article(figure, "fig_latency_paths", ARTICLE_INGENIEUR)


def article_snr() -> Path:
    rows = _rows(EXP3_PI)
    labels = ["clean" if r["condition"] == "clean" else f"{r['snr_db']} dB" for r in rows]
    x = list(range(len(rows)))
    panels = (("crr", "Command recognition rate", 1.0),
              ("wer", "Word error rate", 0.45))
    with plt.rc_context(ARTICLE_RC):
        figure, all_axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.0), sharex=True)
        for axes, (key, label, top) in zip(all_axes, panels):
            value = [float(r[key]) for r in rows]
            lo = [float(r[f"{key}_lo"]) for r in rows]
            hi = [float(r[f"{key}_hi"]) for r in rows]
            axes.fill_between(x, lo, hi, color=BLUE, alpha=0.18, linewidth=0, zorder=2,
                              label="95% bootstrap CI")
            axes.plot(x, value, color=BLUE, marker="o", markersize=3, linewidth=1.2, zorder=3,
                      label=label.lower())
            for xi, v in zip(x, value):
                axes.text(xi, v + top * 0.03, f"{v:.3f}", ha="center", va="bottom",
                          fontsize=5.5, color="0.15")
            axes.set_ylabel(label)
            axes.set_ylim(0, top)
            _article_axes(axes)
        crr = all_axes[0]
        crr.axhline(CRR_CLEAN_TARGET, color=INK, linewidth=0.8, linestyle="--", zorder=1)
        crr.text(5.35, CRR_CLEAN_TARGET + 0.015, "clean target 0.80", ha="right", fontsize=5.8,
                 color="0.15")
        crr.axhline(CRR_10DB_TARGET, color=INK, linewidth=0.8, linestyle=":", zorder=1)
        crr.text(5.35, CRR_10DB_TARGET + 0.015, "10 dB target 0.65", ha="right", fontsize=5.8,
                 color="0.15")
        crr.legend(loc="lower left", frameon=False, handlelength=1.2)
        all_axes[1].set_xticks(x)
        all_axes[1].set_xticklabels(labels)
        all_axes[1].set_xlabel("Rotor-noise condition (signal-to-noise ratio)")
        all_axes[1].set_xlim(-0.3, 5.4)
        figure.align_ylabels(all_axes)
        figure.tight_layout(h_pad=0.5)
        return _save_article(figure, "fig_snr", ARTICLE_INGENIEUR)


def article_formation_boxes() -> Path:
    rows = _rows(EXP4)
    panels = (("convergence_seconds", "Convergence time (s, simulated)", float),
              ("clamp_activations", "Clamp interventions per trial", int))
    with plt.rc_context(ARTICLE_RC):
        figure, all_axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 2.0))
        for axes, (key, label, kind) in zip(all_axes, panels):
            data = [[kind(r[key]) for r in rows if r["shape"] == shape] for shape in FORMATIONS]
            axes.boxplot(data, widths=0.5, patch_artist=True,
                         boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE, linewidth=0.8),
                         medianprops=dict(color="#104281", linewidth=1.2),
                         whiskerprops=dict(color=BLUE, linewidth=0.8),
                         capprops=dict(color=BLUE, linewidth=0.8),
                         flierprops=dict(marker="o", markersize=2.5, markerfacecolor="white",
                                         markeredgecolor=BLUE, markeredgewidth=0.6))
            axes.set_xticks([1, 2, 3])
            axes.set_xticklabels(FORMATIONS)
            axes.set_ylabel(label)
            _article_axes(axes)
        all_axes[0].set_ylim(0, 4.8)
        all_axes[1].set_ylim(bottom=-0.5)
        figure.tight_layout(w_pad=1.2)
        return _save_article(figure, "fig_formation_boxes", ARTICLE_INGENIEUR)


def article_ingenieur() -> list[Path]:
    return [article_keyword_curve(), article_latency_paths(), article_snr(),
            article_formation_boxes()]


def run() -> int:
    if "--article" in sys.argv[1:]:
        for path in article_master() + article_ingenieur():
            print(f"wrote {_display(path)}")
        return 0
    for path in figure_keyword_curve():
        print(f"wrote {_display(path)}")
    plottable, unmeasured = _points()
    for path in figure2():
        print(f"wrote {_display(path)}")
    print()
    for point in sorted(plottable, key=lambda p: p["p95_ms"]):
        frontier = "frontier " if point in pareto_frontier(plottable) else "dominated"
        print(f"  {frontier} {point['model']:24s} {point['quant']:7s} "
              f"EM {point['em']:.4f}  p95 {point['p95_ms']:8.2f} ms  "
              f"RSS {point['rss_mb'] / 1024:.2f} GiB")
    for point in unmeasured:
        print(f"  UNMEASURED {point['model']:24s} {point['quant']:7s} "
              f"EM {point['em']:.4f}  p95 -- (Exp-1 covers Q4_K_M only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
