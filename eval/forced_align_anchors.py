"""Forced-alignment keyword anchors for the reflex path (B-M2, P15).

The ingenieur article times the reflex path from keyword onset and offset, and both anchors
come from energy rules: the capture tool's speech bounds on the 40 recorded takes
(`session_meta.json`, 20 ms frames, so `kw_onset_s`/`kw_offset_s` in Exp-2) and
`eval/closed_loop.py`'s `find_onset` on the closed-loop captures (10 ms levels, floor + 12 dB).
This script replaces them with forced alignments of the known phrase ("swarm hold" or
"swarm abort") and re-states the latencies against them.

Two aligners, both offline on the workstation:

- *MFA* (Montreal Forced Aligner 3, MIT; `english_us_arpa` acoustic model and dictionary,
  CC BY 4.0): HMM-GMM, 10 ms frames. The anchor the latencies are re-stated against.
- *CTC* (torchaudio `forced_align` with `WAV2VEC2_ASR_BASE_960H`, BSD-2 code, MIT weights):
  20 ms frames. A second, independent estimate, so the aligners' disagreement bounds the anchor.

Each aligner is first checked on the 575 synthetic test clips whose keyword start and end
sample are known (the "exact" anchor of `eval/wake_delay_curve.py`), then run on the 40
recorded takes and on a window around every closed-loop stop phrase with a measured latency
(the energy region found by `find_onset`, padded by 0.4 s each side).

What is re-anchored, and what is not: the budget stays 150 ms from the keyword offset, defined
as the end of the spoken phrase (the end of "hold"/"abort"); only the estimate of where that end
lies changes. Exp-2's latencies are shifted per take by (energy anchor - aligned anchor), exact
because Exp-2 stored each publication against `t0 + kw_*_s`. The closed loop is re-run through
`closed_loop.analyse_run` with the aligned onset in place of the energy onset; the energy
region still decides "onset ambiguous", so no take changes outcome class.

Usage (the default environment needs numpy, scipy and soundfile; MFA and torch live elsewhere):

    python eval/forced_align_anchors.py prepare
    python eval/forced_align_anchors.py mfa --mfa-bin ~/miniconda3/envs/mfa/bin/mfa
    <python with torch+torchaudio> eval/forced_align_anchors.py ctc
    python eval/forced_align_anchors.py report

Writes `results/forced_align_anchors.md` and `results/forced_align_anchors.json`; the work
directory (`--work`, default `results/forced_align_work/`) holds the cut audio and raw alignments.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

WORK = REPO / "results" / "forced_align_work"
OUT_MD = REPO / "results" / "forced_align_anchors.md"
OUT_JSON = REPO / "results" / "forced_align_anchors.json"
RATE = 16_000
PAD_WINDOW_S = 0.4
BUDGET_OFFSET_MS = 150.0
BUDGET_ONSET_MS = 850.0
VOICED = 1e-4                      # train_wake.keyword_spans' threshold for the exact end
KEYWORD = {"swarm_hold": "hold", "swarm_abort": "abort", "hover": "hold", "abort": "abort"}


# ----------------------------------------------------------------------------- audio io


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.asarray(audio, dtype="<i2").tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def to_int16(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio
    return np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)


# ----------------------------------------------------------------------------- prepare


def synthetic_items() -> list[dict]:
    """The 575 test clips with an exact keyword end, and their exact start and end (s)."""
    import soundfile as sf

    from train.train_wake import MANIFEST, keyword_spans

    clips = json.loads(MANIFEST.read_text())["clips"]
    spans = keyword_spans(clips)
    # Exact start: first voiced sample of the clean render, by the same rule as the exact end.
    lead: dict[str, float] = {}
    for clip in clips:
        if clip["label"] != "neg" and clip["rir"] is None and clip["noise_key"] is None \
                and clip["base"] not in lead:
            audio, _ = sf.read(REPO / clip["path"])
            voiced = np.nonzero(np.abs(audio) > VOICED)[0]
            lead[clip["base"]] = voiced[0] / RATE - clip["keyword_offset_s"]
    items = []
    for clip in clips:
        if clip["split"] != "test" or clip["label"] == "neg" or not spans[clip["path"]][2]:
            continue
        name = Path(clip["path"]).stem
        items.append({"id": f"syn/{name}", "set": "synthetic", "src": clip["path"],
                      "phrase": clip["phrase"], "speaker": "syn_" + clip["voice"].replace("-", "_"),
                      "clean": clip["rir"] is None and clip["noise_key"] is None,
                      "exact_onset_s": clip["keyword_offset_s"] + lead[clip["base"]],
                      "exact_offset_s": spans[clip["path"]][1], "window_start": 0})
    return items


def real_items() -> list[dict]:
    from eval.wake_real import META, PROMPTS, TAKES

    meta = json.loads(META.read_text())["takes"]
    items = []
    for line in PROMPTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        take, phrase = line.split("\t")
        m = meta[take]
        items.append({"id": f"real/{take}", "set": "real", "take": take,
                      "src": str((TAKES / f"{take}.wav").relative_to(REPO)), "phrase": phrase.strip(),
                      "speaker": "author", "window_start": 0,
                      # Exp-2's anchors without its 1 s pad (eval/exp2.py keyword_trials).
                      "energy_onset_s": m["lead_seconds"],
                      "energy_offset_s": m["seconds"] - m["tail_seconds"]})
    return items


def region_end(audio: np.ndarray, trigger: int, onset: int) -> int:
    """End sample of the energy region that starts at `onset`, with find_onset's floor and gap."""
    from eval import closed_loop as cl

    lo = max(0, trigger - int(cl.FLOOR_S * RATE))
    floor = float(np.percentile(cl.levels_db(audio[lo:trigger]), 10))
    lev = cl.levels_db(audio[onset:onset + int(cl.SEARCH_S * RATE)])
    active = lev > floor + cl.RISE_DB
    gap = int(cl.GAP_S * RATE / cl.LEVEL_WIN)
    end, silent = 0, 0
    for i, a in enumerate(active):
        if a:
            end, silent = i + 1, 0
        else:
            silent += 1
            if silent > gap:
                break
    return onset + end * cl.LEVEL_WIN


def closed_loop_calls(raw: Path) -> dict[str, list[dict]]:
    """Every find_onset call closed_loop.analyse_run makes, per run, with the energy answer."""
    from eval import closed_loop as cl

    calls: dict[str, list[dict]] = {}
    original = cl.find_onset

    def recorder(audio, trigger_sample, **kw):
        found = original(audio, trigger_sample, **kw)
        entry = {"trigger": int(trigger_sample), "energy": None if found is None else int(found[0]),
                 "region_s": None if found is None else float(found[1])}
        if found is not None:
            entry["region_end"] = int(region_end(audio, trigger_sample, found[0]))
        calls[current["run"]].append(entry)
        return found

    current = {"run": ""}
    cl.find_onset = recorder
    try:
        for run in sorted(raw.iterdir()):
            if (run / "meta.json").is_file():
                current["run"] = run.name
                calls[run.name] = []
                _, takes, _ = cl.analyse_run(run)
                # analyse_run calls find_onset only for takes that reach it, in order.
                reaching = [t for t in takes if t.outcome not in ("lost on the link", "outside the trace")]
                assert len(reaching) == len(calls[run.name]), run
                for t, c in zip(reaching, calls[run.name]):
                    c.update({"intent": t.intent, "path": t.path, "seq": t.seq})
    finally:
        cl.find_onset = original
    return calls


def closed_loop_items(raw: Path) -> list[dict]:
    from eval import closed_loop as cl

    items = []
    for run, calls in closed_loop_calls(raw).items():
        audio, _ = cl.load_audio(raw / run)
        audio = to_int16(audio)
        for c in calls:
            if c["energy"] is None:
                continue
            start = max(0, c["energy"] - int(PAD_WINDOW_S * RATE))
            stop = min(audio.size, max(c["region_end"], c["trigger"]) + int(PAD_WINDOW_S * RATE))
            items.append({"id": f"cl/{run}_{c['path']}_{c['seq']}", "set": "closed_loop", "run": run,
                          "path": c["path"], "seq": c["seq"], "intent": c["intent"],
                          "trigger": c["trigger"], "phrase": f"swarm {KEYWORD[c['intent']]}",
                          "speaker": "author", "window_start": start, "window_stop": stop,
                          "energy_onset_s": (c["energy"] - start) / RATE,
                          "energy_region_s": c["region_s"]})
    return items


def prepare(work: Path, raw: Path) -> int:
    import soundfile as sf

    from eval import closed_loop as cl

    items = synthetic_items() + real_items() + closed_loop_items(raw)
    corpus = work / "corpus"
    audio_cache: dict[str, np.ndarray] = {}
    for it in items:
        if it["set"] == "closed_loop":
            if it["run"] not in audio_cache:
                audio_cache[it["run"]] = to_int16(cl.load_audio(raw / it["run"])[0])
            audio = audio_cache[it["run"]][it["window_start"]:it["window_stop"]]
        else:
            audio, rate = sf.read(REPO / it["src"], dtype="int16")
            assert rate == RATE and audio.ndim == 1, it["src"]
        stem = it["id"].split("/", 1)[1]
        wav = corpus / it["speaker"] / f"{stem}.wav"
        write_wav(wav, audio)
        (wav.with_suffix(".lab")).write_text(it["phrase"] + "\n", encoding="utf-8")
        it["wav"] = str(wav.relative_to(work))
    (work / "items.json").write_text(json.dumps(items, indent=1) + "\n", encoding="utf-8")
    counts = {s: sum(i["set"] == s for i in items) for s in ("synthetic", "real", "closed_loop")}
    print(f"prepared {counts} under {corpus}")
    return 0


# ----------------------------------------------------------------------------- aligners


def run_mfa(work: Path, mfa_bin: Path) -> int:
    env = dict(os.environ, PATH=f"{mfa_bin.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    for model in ("acoustic", "dictionary"):
        subprocess.run([str(mfa_bin), "model", "download", model, "english_us_arpa"], env=env, check=True)
    out = work / "mfa"
    subprocess.run([str(mfa_bin), "align", str(work / "corpus"), "english_us_arpa", "english_us_arpa",
                    str(out), "--clean", "--overwrite", "--output_format", "json",
                    "--beam", "100", "--retry_beam", "400", "-j", str(os.cpu_count() or 4)],
                   env=env, check=True)
    items = json.loads((work / "items.json").read_text())
    words: dict[str, list] = {}
    for it in items:
        path = out / Path(it["wav"]).relative_to("corpus").with_suffix(".json")
        if not path.is_file():
            continue
        tiers = json.loads(path.read_text())["tiers"]
        entries = tiers["words"]["entries"]
        words[it["id"]] = [[float(s), float(e), str(w)] for s, e, w in entries if str(w).strip()]
    (work / "mfa_words.json").write_text(json.dumps(words, indent=1) + "\n", encoding="utf-8")
    print(f"mfa: {len(words)}/{len(items)} aligned")
    return 0


def run_ctc(work: Path) -> int:
    import torch
    import torchaudio
    import torchaudio.functional as F

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = bundle.get_model().to(device).eval()
    labels = bundle.get_labels()
    index = {c: i for i, c in enumerate(labels)}
    items = json.loads((work / "items.json").read_text())
    words: dict[str, list] = {}
    for it in items:
        audio = read_wav(work / it["wav"]).astype(np.float32) / 32768.0
        with torch.inference_mode():
            emission, _ = model(torch.from_numpy(audio)[None].to(device))
            emission = torch.log_softmax(emission, dim=-1).cpu()
        transcript = it["phrase"].upper().split()
        tokens = [index[c] for w in transcript for c in w]
        aligned, scores = F.forced_align(emission, torch.tensor([tokens], dtype=torch.int32), blank=0)
        spans = F.merge_tokens(aligned[0], scores[0].exp())
        stride = audio.size / emission.size(1) / RATE        # 320 samples, 20 ms
        out, k = [], 0
        for w in transcript:
            group = spans[k:k + len(w)]
            k += len(w)
            out.append([group[0].start * stride, group[-1].end * stride, w.lower()])
        words[it["id"]] = out
    (work / "ctc_words.json").write_text(json.dumps(words, indent=1) + "\n", encoding="utf-8")
    print(f"ctc ({device}): {len(words)}/{len(items)} aligned")
    return 0


# ----------------------------------------------------------------------------- analysis


def anchors(words: list | None, phrase: str) -> dict | None:
    """Onset of "swarm", and onset and offset of the keyword, from an aligner's word list."""
    if not words:
        return None
    expected = phrase.split()
    found = [w for w in words if w[2].lower() in expected]
    if [w[2].lower() for w in found] != expected:
        return None
    return {"swarm_onset": found[0][0], "swarm_offset": found[0][1],
            "kw_onset": found[1][0], "kw_offset": found[1][1]}


def stats_ms(diffs: list[float]) -> dict:
    d = np.asarray([x for x in diffs if math.isfinite(x)])
    if d.size == 0:
        return {"n": 0}
    i = int(np.argmax(np.abs(d)))
    return {"n": int(d.size), "median": float(np.median(d)), "median_abs": float(np.median(np.abs(d))),
            "p95_abs": float(np.percentile(np.abs(d), 95)), "largest": float(d[i]),
            "within_20": int(np.sum(np.abs(d) <= 20)), "within_50": int(np.sum(np.abs(d) <= 50))}


def pct_cell(values: list[float], q: float) -> dict:
    from eval.stats import nearest_rank, order_statistic_ci

    lo, hi = order_statistic_ci(values, q)
    return {"value": float(nearest_rank(values, q)), "lo": None if math.isinf(lo) else float(lo),
            "hi": None if math.isinf(hi) else float(hi)}


def exp2_rows(real: dict[str, dict], aligner: str) -> dict:
    """Exp-2 pass-0 latencies, energy and aligned, per condition (the take as the unit)."""
    from eval.exp2 import algorithmic_ms, first_own_event

    out = {}
    for condition in ("idle", "loaded"):
        rows = [json.loads(line) for line in (REPO / "results" / "exp2_raw" / f"{condition}.jsonl")
                .read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r["pass"] == 0]
        cols: dict[str, list[float]] = {k: [] for k in (
            "offset_energy", "offset_aligned", "onset_energy", "onset_aligned",
            "algo_energy", "algo_aligned", "swarm_offset_aligned")}
        per_take = []
        for r in rows:
            e = first_own_event(r)
            a = real[r["take"]].get(aligner)
            if e is None or a is None:
                continue
            d_off = (r["kw_offset_s"] - (1.0 + a["kw_offset"])) * 1000     # Exp-2 pads 1 s
            d_on = (r["kw_onset_s"] - (1.0 + a["swarm_onset"])) * 1000
            cols["offset_energy"].append(e["latency_offset_ms"])
            cols["offset_aligned"].append(e["latency_offset_ms"] + d_off)
            cols["onset_energy"].append(e["latency_onset_ms"])
            cols["onset_aligned"].append(e["latency_onset_ms"] + d_on)
            cols["algo_energy"].append(algorithmic_ms(r, e))
            cols["algo_aligned"].append(algorithmic_ms(r, e) + d_off)
            cols["swarm_offset_aligned"].append(
                e["latency_offset_ms"] + (r["kw_offset_s"] - (1.0 + a["swarm_offset"])) * 1000)
            per_take.append({"take": r["take"], "offset_energy": e["latency_offset_ms"],
                             "offset_aligned": e["latency_offset_ms"] + d_off})
        out[condition] = {k: {"n": len(v), "p50": pct_cell(v, 50), "p95": pct_cell(v, 95),
                              "early": int(sum(x < 0 for x in v)), "min": float(min(v)),
                              "within_budget": int(sum(x <= (BUDGET_ONSET_MS if k.startswith("onset")
                                                             else BUDGET_OFFSET_MS) for x in v))}
                          for k, v in cols.items()}
        out[condition]["worst"] = sorted(per_take, key=lambda t: -t["offset_aligned"])[:3]
    return out


def _pcts(values: list[float]) -> dict:
    return {"n": len(values), "p50": pct_cell(values, 50), "p95": pct_cell(values, 95)}


def baseline_rows(real: dict[str, dict], aligner: str = "mfa") -> dict:
    """The voice-onset hold (P12) and the single path (P12) on the same takes, re-anchored."""
    from eval.single_path import first_stop, raw_path

    def shift(take: str, kind: str) -> float:
        it, a = real[take], real[take][aligner]
        if kind == "onset":
            return (it["energy_onset_s"] - a["swarm_onset"]) * 1000
        return (it["energy_offset_s"] - a["kw_offset"]) * 1000

    out: dict = {}
    rows = [json.loads(line) for line in (REPO / "results" / "voice_onset_raw.jsonl").read_text().splitlines()
            if line.strip()]
    first = [(r["take"], r["events"][0]) for r in rows if r["events"]]
    for kind in ("onset", "offset"):
        key = f"latency_{kind}_ms"
        energy = [e[key] for _, e in first]
        aligned = [e[key] + shift(t, kind) for t, e in first]
        out[f"voice_onset/{kind}"] = {"energy": _pcts(energy), aligner: _pcts(aligned),
                                      "early_energy": int(sum(x < 0 for x in energy)),
                                      "early_aligned": int(sum(x < 0 for x in aligned))}
    from eval.ordering_replay import first_own_event as own

    rows = [json.loads(line) for line in (REPO / "results" / "ordering_replay_raw.jsonl").read_text()
            .splitlines() if line.strip()]
    for mode in ("acceptance", "onset"):
        evs = [(r["take"], e) for r in rows if r["mode"] == mode and (e := own(r)) is not None]
        out[f"ordering/{mode}/reflex_offset"] = {
            "energy": _pcts([e["latency_offset_ms"] for _, e in evs]),
            aligner: _pcts([e["latency_offset_ms"] + shift(t, "offset") for t, e in evs])}
    for host in ("drone-pi", "kahia-tayeb"):
        path = raw_path(host)
        if not path.is_file():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        stops = [(r["take"], first_stop(r)) for r in rows]
        stops = [(t, s) for t, s in stops if s is not None]
        out[f"single_path/{host}/offset_to_applied"] = {
            "energy": _pcts([s["latency_offset_ms"] for _, s in stops]),
            aligner: _pcts([s["latency_offset_ms"] + shift(t, "offset") for t, s in stops])}
        ends = [(r["take"], r["segments"][0]["endpoint_after_offset_ms"]) for r in rows if r["segments"]]
        out[f"single_path/{host}/endpoint_after_offset"] = {
            "energy": _pcts([v for _, v in ends]), aligner: _pcts([v + shift(t, "offset") for t, v in ends])}
    return out


def closed_loop_aligned(raw: Path, items: list[dict], aligner: str):
    """closed_loop.analyse_run with the aligned onset of "swarm" in place of the energy onset."""
    from eval import closed_loop as cl

    table = {(it["run"], it["trigger"]): it for it in items}
    original = cl.find_onset
    current = {"run": ""}

    def aligned(audio, trigger_sample, **kw):
        found = original(audio, trigger_sample, **kw)
        it = table.get((current["run"], int(trigger_sample)))
        if found is None or it is None or it.get(aligner) is None:
            return found if it is None else None
        onset = it["window_start"] + int(round(it[aligner]["swarm_onset"] * RATE))
        return onset, found[1]          # the energy region still decides "onset ambiguous"

    cl.find_onset = aligned
    try:
        runs = []
        for run in sorted(raw.iterdir()):
            if (run / "meta.json").is_file():
                current["run"] = run.name
                runs.append(cl.analyse_run(run))
    finally:
        cl.find_onset = original
    return runs


def closed_loop_summary(runs) -> dict:
    from eval.closed_loop import CONDITIONS, INTENTS

    from eval.stats import order_statistic_ci

    out = {}
    takes = [t for _, ts, _ in runs for t in ts]
    # The article pools the parse-path stops over both conditions ("both").
    for condition in (*CONDITIONS, "both"):
        for intent in INTENTS:
            for path in ("reflex", "parse"):
                if condition == "both" and path == "reflex":
                    continue
                ok = [t for t in takes if condition in (t.condition, "both") and t.intent == intent
                      and t.path == path and t.outcome == "measured"]
                if not ok:
                    continue
                cell = {"n": len(ok)}
                for key in ("onset_to_apply_ms", "onset_to_rest_ms", "distance_from_onset_m",
                            "onset_to_publish_ms"):
                    v = [getattr(t, key) for t in ok if math.isfinite(getattr(t, key))]
                    lo, hi = order_statistic_ci(v, 50)
                    cell[key] = {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v)),
                                 "lo": None if math.isinf(lo) else float(lo),
                                 "hi": None if math.isinf(hi) else float(hi)}
                out[f"{condition}/{intent}/{path}"] = cell
    return out


def report(work: Path, raw: Path) -> int:
    from eval import closed_loop as cl

    items = json.loads((work / "items.json").read_text())
    aligned = {"mfa": json.loads((work / "mfa_words.json").read_text()),
               "ctc": json.loads((work / "ctc_words.json").read_text())}
    for it in items:
        for name, words in aligned.items():
            it[name] = anchors(words.get(it["id"]), it["phrase"])

    record: dict = {"counts": {}, "validation": {}, "real": {}, "closed_loop": {}}
    # 1. Validation on the exact anchor.
    for name in aligned:
        for subset, keep in (("clean", True), ("augmented", False)):
            syn = [it for it in items if it["set"] == "synthetic" and it["clean"] == keep]
            ok = [it for it in syn if it[name]]
            record["validation"][f"{name}/{subset}"] = {
                "n": len(syn), "aligned": len(ok),
                "onset": stats_ms([(it[name]["swarm_onset"] - it["exact_onset_s"]) * 1000 for it in ok]),
                "offset": stats_ms([(it[name]["kw_offset"] - it["exact_offset_s"]) * 1000 for it in ok])}
    # 1b. Where each offset sits on the clean clips' decay: last 10 ms window within X dB of the peak.
    import soundfile as sf

    decay = {db: {"mfa": [], "exact": []} for db in (26, 40, 60)}
    for it in items:
        if it["set"] != "synthetic" or not it["clean"] or not it["mfa"]:
            continue
        x, _ = sf.read(REPO / it["src"])
        n = x.size // 160
        lev = 10 * np.log10(np.mean(x[: n * 160].reshape(n, 160) ** 2, axis=1) + 1e-20)
        for db in decay:
            end = (np.flatnonzero(lev > lev.max() - db)[-1] + 1) * 0.010
            decay[db]["mfa"].append((it["mfa"]["kw_offset"] - end) * 1000)
            decay[db]["exact"].append((it["exact_offset_s"] - end) * 1000)
    record["decay"] = {str(db): {k: stats_ms(v) for k, v in d.items()} for db, d in decay.items()}
    # 2. Recorded takes and closed-loop phrases against the energy anchors and each other.
    real = [it for it in items if it["set"] == "real"]
    closed_all = [it for it in items if it["set"] == "closed_loop"]
    # A region longer than closed_loop.MAX_REGION_S merged the phrase with earlier speech; its take
    # is excluded as "onset ambiguous" there, so its energy onset is not compared here either.
    closed = [it for it in closed_all if it["energy_region_s"] <= cl.MAX_REGION_S]
    record["counts"] = {"synthetic": sum(it["set"] == "synthetic" for it in items), "real": len(real),
                        "closed_loop": len(closed_all), "closed_loop_unambiguous": len(closed)}
    for name in aligned:
        record["real"][name] = {
            "aligned": sum(bool(it[name]) for it in real), "n": len(real),
            "onset_minus_energy": stats_ms([(it[name]["swarm_onset"] - it["energy_onset_s"]) * 1000
                                            for it in real if it[name]]),
            "offset_minus_energy": stats_ms([(it[name]["kw_offset"] - it["energy_offset_s"]) * 1000
                                             for it in real if it[name]])}
        for phrase in ("swarm hold", "swarm abort"):
            record["real"][name][f"offset_minus_energy/{phrase}"] = stats_ms(
                [(it[name]["kw_offset"] - it["energy_offset_s"]) * 1000
                 for it in real if it[name] and it["phrase"] == phrase])
        record["closed_loop"][name] = {
            "aligned": sum(bool(it[name]) for it in closed_all), "n": len(closed_all),
            "onset_minus_energy": stats_ms([(it[name]["swarm_onset"] - it["energy_onset_s"]) * 1000
                                            for it in closed if it[name]])}
    both = lambda pool, key: stats_ms([(it["ctc"][key] - it["mfa"][key]) * 1000  # noqa: E731
                                       for it in pool if it["mfa"] and it["ctc"]])
    record["agreement"] = {"real_onset": both(real, "swarm_onset"), "real_offset": both(real, "kw_offset"),
                           "closed_loop_onset": both(closed, "swarm_onset")}
    closed = closed_all
    record["per_take"] = [{"take": it["take"], "phrase": it["phrase"],
                           "energy_onset_s": it["energy_onset_s"], "energy_offset_s": it["energy_offset_s"],
                           "mfa": it["mfa"], "ctc": it["ctc"]} for it in real]
    record["per_phrase"] = [{k: it.get(k) for k in ("id", "run", "path", "seq", "intent", "window_start",
                                                    "energy_onset_s", "energy_region_s", "mfa", "ctc")}
                            for it in closed]
    # 3. Re-stated latencies.
    by_take = {it["take"]: it for it in real}
    record["exp2"] = {name: exp2_rows(by_take, name) for name in aligned}
    record["baselines"] = baseline_rows(by_take, "mfa")
    record["closed_loop_latency"] = {"energy": closed_loop_summary([cl.analyse_run(r) for r in sorted(raw.iterdir())
                                                                    if (r / "meta.json").is_file()])}
    runs_mfa = None
    for name in aligned:
        runs = closed_loop_aligned(raw, closed, name)
        record["closed_loop_latency"][name] = closed_loop_summary(runs)
        if name == "mfa":
            runs_mfa = runs
    OUT_JSON.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    OUT_MD.write_text(markdown(record, cl.report(runs_mfa)), encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(REPO)} and {OUT_JSON.relative_to(REPO)}")
    return 0


# ----------------------------------------------------------------------------- markdown


def _f(x, nd=0) -> str:
    return "--" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:,.{nd}f}"


def _cell(c: dict) -> str:
    return f"{_f(c['value'])} [{_f(c['lo'])}, {_f(c['hi'])}]"


def _diff_row(label: str, s: dict) -> str:
    if not s.get("n"):
        return f"| {label} | 0 | -- | -- | -- | -- | -- |"
    return (f"| {label} | {s['n']} | {s['median']:+.0f} | {s['median_abs']:.0f} | {s['p95_abs']:.0f} | "
            f"{s['largest']:+.0f} | {s['within_20']}/{s['n']}, {s['within_50']}/{s['n']} |")


DIFF_HEAD = ["| Comparison | n | median (ms) | median abs | p95 abs | largest | within 20 / 50 ms |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]


def markdown(rec: dict, closed_loop_md: str) -> str:
    L = ["# Forced-alignment keyword anchors (B-M2, P15)", "",
         "Generated by `python eval/forced_align_anchors.py report`. Do not edit by hand.", "",
         "Anchors: *energy* is the anchor the article used (capture tool's 20 ms speech bounds on the "
         "recorded takes; `closed_loop.find_onset`'s 10 ms floor + 12 dB rule on the closed-loop "
         "captures). *MFA* is Montreal Forced Aligner 3 with `english_us_arpa` (10 ms frames); *CTC* is "
         "torchaudio `forced_align` on `WAV2VEC2_ASR_BASE_960H` (20 ms frames). Onset is the start of "
         "\"swarm\"; offset is the end of \"hold\"/\"abort\". Differences are aligner minus reference, "
         "in ms; positive means the aligner puts the boundary later.", "",
         "The budget is unchanged: 150 ms at p95 from the keyword offset (the end of the spoken phrase) "
         "and 850 ms from its onset. Only the estimate of where the phrase starts and ends changes.", "",
         "## 1. Aligner check on the exact anchor (synthetic test split)", "",
         "Exact start and end: first and last sample above 1e-4 in the clean Piper render "
         "(`train_wake.keyword_spans`). Augmented clips carry room reverberation and/or noise.", ""]
    L += DIFF_HEAD
    for key, v in rec["validation"].items():
        name, subset = key.split("/")
        L.append(_diff_row(f"{name.upper()} onset, {subset} ({v['aligned']}/{v['n']} aligned)", v["onset"]))
        L.append(_diff_row(f"{name.upper()} offset, {subset}", v["offset"]))
    L += ["", "Where the offsets sit on the clean clips' decay: offset minus the end of the last 10 ms "
          "window within X dB of the clip's loudest window (the capture tool's energy rule on the "
          "recorded takes ends at 26 dB below peak, or 12 dB above the floor if higher).", ""]
    L += DIFF_HEAD
    for db, d in rec["decay"].items():
        L.append(_diff_row(f"MFA offset - end at peak - {db} dB", d["mfa"]))
        L.append(_diff_row(f"Exact offset - end at peak - {db} dB", d["exact"]))
    c = rec["counts"]
    L += ["", "## 2. Recorded takes (40) and closed-loop stop phrases against the energy anchors", "",
          f"Closed loop: {c['closed_loop']} stop phrases with a measured latency (reflex detections and "
          "parse-path stops after a reflex miss), each aligned in its energy region padded by "
          f"{PAD_WINDOW_S} s. Differences use the {c['closed_loop_unambiguous']} whose energy region is at "
          "most `MAX_REGION_S`; the others merged with earlier speech and are excluded as \"onset "
          "ambiguous\" by `closed_loop.py`.", ""]
    L += DIFF_HEAD
    for name in ("mfa", "ctc"):
        r, c = rec["real"][name], rec["closed_loop"][name]
        L.append(_diff_row(f"{name.upper()} - energy, onset, recorded takes ({r['aligned']}/{r['n']})",
                           r["onset_minus_energy"]))
        L.append(_diff_row(f"{name.upper()} - energy, offset, recorded takes", r["offset_minus_energy"]))
        for phrase in ("swarm hold", "swarm abort"):
            L.append(_diff_row(f"{name.upper()} - energy, offset, \"{phrase}\" only",
                               r[f"offset_minus_energy/{phrase}"]))
        L.append(_diff_row(f"{name.upper()} - energy, onset, closed loop ({c['aligned']}/{c['n']})",
                           c["onset_minus_energy"]))
    a = rec["agreement"]
    L.append(_diff_row("CTC - MFA, onset, recorded takes", a["real_onset"]))
    L.append(_diff_row("CTC - MFA, offset, recorded takes", a["real_offset"]))
    L.append(_diff_row("CTC - MFA, onset, closed loop", a["closed_loop_onset"]))
    L += ["", "## 3. Reflex path (Exp-2, pass 0, `drone-pi`) re-anchored", "",
          "Each detection's latency shifted by (energy anchor - aligned anchor) of its take. "
          "\"Algorithmic\" is the spotter delay plus frame quantisation (Pi compute excluded).", "",
          "| Condition | Measure | Anchor | n | p50 [95% CI] (ms) | p95 [95% CI] (ms) | before anchor | within budget |",
          "| :--- | :--- | :--- | ---: | :--- | :--- | ---: | ---: |"]
    labels = [("offset", "from keyword offset"), ("algo", "algorithmic, from offset"),
              ("onset", "from keyword onset")]
    for condition in ("idle", "loaded"):
        for key, label in labels:
            for anchor in ("energy", "mfa", "ctc"):
                col = f"{key}_{'energy' if anchor == 'energy' else 'aligned'}"
                c = rec["exp2"]["mfa" if anchor == "energy" else anchor][condition][col]
                budget = "" if key == "algo" else f"{c['within_budget']}/{c['n']}"
                early = f"{c['early']}/{c['n']}" + (f" (by up to {-c['min']:.0f})" if c["early"] else "")
                L.append(f"| {condition} | {label} | {anchor} | {c['n']} | {_cell(c['p50'])} | "
                         f"{_cell(c['p95'])} | {early} | {budget} |")
    for name in ("mfa",):
        w = rec["exp2"][name]["idle"]["worst"]
        L += ["", "Slowest takes from the aligned offset (idle, MFA): " + ", ".join(
            f"{t['take']} {t['offset_aligned']:.0f} ms (energy {t['offset_energy']:.0f})" for t in w) + "."]
    L += ["", "### Baselines on the same takes (P12), re-anchored to MFA", "",
          "| Measure | n | energy p50 [95% CI] | energy p95 [95% CI] | MFA p50 [95% CI] | MFA p95 [95% CI] |",
          "| :--- | ---: | :--- | :--- | :--- | :--- |"]
    names = {"voice_onset/onset": "Voice-onset hold, from keyword onset",
             "voice_onset/offset": "Voice-onset hold, from keyword offset",
             "ordering/acceptance/reflex_offset": "Both paths, number at acceptance: reflex from offset",
             "ordering/onset/reflex_offset": "Both paths, number at onset: reflex from offset",
             "single_path/drone-pi/endpoint_after_offset":"Single path, endpointing after offset (`drone-pi`)",
             "single_path/drone-pi/offset_to_applied": "Single path, offset to applied (`drone-pi`)",
             "single_path/kahia-tayeb/endpoint_after_offset": "Single path, endpointing after offset (workstation)",
             "single_path/kahia-tayeb/offset_to_applied": "Single path, offset to applied (workstation)"}
    for key, label in names.items():
        b = rec["baselines"].get(key)
        if b is None:
            continue
        e, m = b["energy"], b["mfa"]
        L.append(f"| {label} | {e['n']} | {_cell(e['p50'])} | {_cell(e['p95'])} | {_cell(m['p50'])} | "
                 f"{_cell(m['p95'])} |")
    vo = rec["baselines"]["voice_onset/onset"]
    L += ["", f"Voice-onset holds before the keyword onset: {vo['early_energy']}/{vo['energy']['n']} "
          f"(energy), {vo['early_aligned']}/{vo['mfa']['n']} (MFA)."]
    L += ["", "## 4. Closed loop re-anchored (median and range, ms)", "",
          "Cells are condition/intent/path; \"both\" pools online and offline, as the article's table does "
          "for the parse path.", "",
          "| Cell | n | onset to applied: energy | MFA | CTC | onset to rest: energy | MFA | CTC |",
          "| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- |"]
    cl = rec["closed_loop_latency"]
    for cell, e in cl["energy"].items():
        def mr(src, key):
            c = cl[src].get(cell, {}).get(key)
            return "--" if c is None else f"{c['median']:,.0f} ({c['min']:,.0f}--{c['max']:,.0f})"
        L.append(f"| {cell} | {e['n']} | {mr('energy', 'onset_to_apply_ms')} | {mr('mfa', 'onset_to_apply_ms')} | "
                 f"{mr('ctc', 'onset_to_apply_ms')} | {mr('energy', 'onset_to_rest_ms')} | "
                 f"{mr('mfa', 'onset_to_rest_ms')} | {mr('ctc', 'onset_to_rest_ms')} |")
    L += ["", "MFA, medians with 95% order-statistic intervals (the article's Table `tab:closed` columns):", "",
          "| Cell | n | onset to applied (s) | onset to rest (s) | distance from onset (m) |",
          "| :--- | ---: | :--- | :--- | :--- |"]
    for cell, c in cl["mfa"].items():
        def ci(key, scale, nd):
            v = c[key]
            if v["lo"] is None or v["hi"] is None or c["n"] < 9:
                return f"{v['median'] * scale:.{nd}f}"
            return f"{v['median'] * scale:.{nd}f} [{v['lo'] * scale:.{nd}f}, {v['hi'] * scale:.{nd}f}]"
        L.append(f"| {cell} | {c['n']} | {ci('onset_to_apply_ms', 1e-3, 2)} | {ci('onset_to_rest_ms', 1e-3, 1)} | "
                 f"{ci('distance_from_onset_m', 1, 2)} |")
    L += ["", "## 5. Closed-loop tables against the MFA onset", "",
          "`eval/closed_loop.py`'s own report, re-run with the MFA onset of \"swarm\" in place of the "
          "energy onset (the energy region still decides \"onset ambiguous\").", ""]
    body = closed_loop_md.split("\n")
    body = [ln for ln in body if not ln.startswith("Generated by")]
    L += ["#" + ln if ln.startswith("###") else ln for ln in body]
    return "\n".join(L).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("prepare", "mfa", "ctc", "report"))
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--raw", type=Path, default=REPO / "results" / "closed_loop_raw")
    ap.add_argument("--mfa-bin", type=Path, default=Path.home() / "miniconda3" / "envs" / "mfa" / "bin" / "mfa")
    args = ap.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    if args.step == "prepare":
        return prepare(args.work, args.raw)
    if args.step == "mfa":
        return run_mfa(args.work, args.mfa_bin)
    if args.step == "ctc":
        return run_ctc(args.work)
    return report(args.work, args.raw)


if __name__ == "__main__":
    raise SystemExit(main())
