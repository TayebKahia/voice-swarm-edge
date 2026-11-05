#!/usr/bin/env python3
"""Branch A under rotor noise --- the condition the keyword spotter exists for.

`results/wake_real_voice.md` scored the spotter on the author's 40 recorded takes in a
quiet room, and its own last bullet says so: it "closes the *synthetic-voice* caveat,
not the speaker or noise ones". The parse path was swept over DREGON rotor noise
(Exp-3); the spotter never was. This script does that sweep for the spotter.

**Mixing is `data/mix_noise.py`'s, unchanged.** SNR is set on the take's active-speech
level (frames within 30 dB of the loudest), only the noise is scaled, one excerpt per
take is reused at every SNR (seed 42 plus the take id), and the noise comes from the
DREGON **evaluation** partition: the last 30% of the hovering flight. `data/wake_corpus.py`
mixed the `aug` partition (first 70%, channel 0) into the spotter's train and val clips,
so the eval partition is the only DREGON audio the heads were not fitted on.

**The noise covers the whole scored window.** Each take is scored as `train_wake.py`
scores its test split, with 1 s of padding either side, but the padding here carries
rotor noise rather than silence: on a hovering vehicle the noise does not start with the
word. The padded signal is mixed as one item; its zero padding is below the active-speech
floor, so the SNR is still the take's.

**Scoring is `eval/wake_real.py`'s, unchanged:** the trained heads (hash-checked), the
deployed threshold from `wake_heads.json`, 80 ms frames through `openwakeword.Model`, a
take detected if its own head ever reaches the threshold, the 2x2 cross-trigger matrix,
and delay from the capture tool's end-of-speech bound. A `clean` row runs the same code
with no noise and must reproduce `results/wake_real_voice.md`.

**False triggers on noise alone** are counted on the same eval partition, played at the
level the noise had in the mixes: for each SNR, noise RMS = (median active-speech RMS of
the 40 takes) / 10^(SNR/20). The primary row is channel 0, the channel training used. The
partition is 13.5 s long, so a per-hour rate from it is nearly uninformative and is
reported with its Poisson interval as such. A secondary row concatenates all eight array
channels of the same 13.5 s: eight microphones on one airframe hear the same rotors, so
they are not independent hours, but channels 1--7 were never in training.

**These mixes are a test set.** The threshold is read, never chosen, here.

Usage:
    python eval/wake_noise.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python eval/wake_noise.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from eval.wake_real import META, TAKES, _check_heads, clopper_pearson, labels

REPO = Path(__file__).resolve().parent.parent
OUT_MD = REPO / "results" / "wake_noise.md"
OUT_JSON = REPO / "results" / "wake_noise.json"

SNR_LEVELS = (20, 10, 5)
SEED = 42                    # data/mix_noise.py's DEFAULT_SEED
DEPLOYED_THRESHOLD = 0.999   # wake_heads.json; asserted, not assumed
PAD_S = 1.0
RATE = 16_000
NFR16_FRR = 0.10
NFR15_FA_PER_HOUR = 1.0


def _to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 32768.0


def _to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def _dregon_eval_channels() -> list[np.ndarray]:
    """Every channel of the DREGON flight, cut to the eval partition exactly as NoiseBank cuts it."""
    from data.mix_noise import NoiseBank, load_mono

    bank = NoiseBank(sources=("dregon",), partition="eval")
    (key, entry), = bank._entries  # one flight
    path = REPO / entry["path"]
    import soundfile as sf

    channels = sf.info(str(path)).channels
    out = []
    for ch in range(channels):
        x = load_mono(path, RATE, channel=ch)
        out.append(x[int(len(x) * NoiseBank.DREGON_AUG_FRACTION):])
    # Channel 0 must be the very samples the mixer draws from.
    if not np.array_equal(out[0], bank._load(key, entry).samples):
        raise SystemExit("channel-0 eval partition differs from NoiseBank's; the cut has drifted")
    return out


def run() -> int:
    from data.mix_noise import NoiseBank, active_speech_rms, mix_one, rms
    from train.train_wake import (CLASSES, _load_int16, _model, _pct, count_events, poisson_ci,
                                  score_positives, stream_scores)

    threshold = _check_heads()
    if threshold != DEPLOYED_THRESHOLD:
        raise SystemExit(f"wake_heads.json threshold is {threshold}, not the deployed {DEPLOYED_THRESHOLD}")

    meta = json.loads(META.read_text())["takes"]
    classes = labels()
    bank = NoiseBank(sources=("dregon",), partition="eval")
    model = _model()
    pad = np.zeros(int(PAD_S * RATE), dtype=np.int16)

    takes = {}
    for take, name in sorted(classes.items()):
        audio = _load_int16(TAKES / f"{take}.wav")
        takes[take] = {"name": name, "padded": np.concatenate([pad, audio, pad]),
                       "kw_end_s": PAD_S + meta[take]["seconds"] - meta[take]["tail_seconds"],
                       "active_rms": active_speech_rms(_to_float(audio), RATE)}

    conditions: dict[str, dict] = {}
    peaks: dict[str, dict] = {}
    for cond in ["clean", *[f"{snr} dB" for snr in SNR_LEVELS]]:
        traces, mix_log = [], {}
        for take, t in takes.items():
            if cond == "clean":
                signal = t["padded"]
            else:
                snr = int(cond.split()[0])
                res = mix_one(_to_float(t["padded"]), snr, bank, take, seed=SEED, sample_rate=RATE)
                signal = _to_int16(res.audio)
                mix_log[take] = {"noise_gain": round(res.noise_gain, 6),
                                 "peak_attenuation": round(res.peak_attenuation, 6)}
            traces.append({"take": take, "label": CLASSES[t["name"]], "kw_end_s": t["kw_end_s"],
                           "trace": stream_scores(model, signal)})

        scored = score_positives(traces, threshold)
        rows = []
        for name in CLASSES:
            mine = [tr for tr in traces if tr["label"] == CLASSES[name]]
            missed = [tr["take"] for tr in mine if not (tr["trace"][name] >= threshold).any()]
            other = next(o for o in CLASSES if o != name)
            k, n = len(missed), len(mine)
            lo, hi = clopper_pearson(k, n)
            d = scored["delay_s"][name]
            rows.append({"class": name, "n": n, "missed": k, "frr": k / n, "frr_ci95": [lo, hi],
                         "fired_own": scored["cross_trigger"][name][name],
                         "fired_other": scored["cross_trigger"][name][other],
                         "fired_other_takes": [tr["take"] for tr in mine
                                               if (tr["trace"][other] >= threshold).any()],
                         "delay_p50_ms": _pct(d, 50) * 1000 if d else None,
                         "delay_p95_ms": _pct(d, 95) * 1000 if d else None,
                         "missed_takes": missed})
        conditions[cond] = {"rows": rows, "mix": mix_log}
        peaks[cond] = {tr["take"]: {n: round(float(tr["trace"][n].max()), 4) for n in CLASSES}
                       for tr in traces}

    # --- noise alone ---------------------------------------------------------
    channels = _dregon_eval_channels()
    ref_level = float(np.median([t["active_rms"] for t in takes.values()]))
    noise_rows = []
    for snr in SNR_LEVELS:
        target = ref_level / (10.0 ** (snr / 20.0))
        for label, chans in (("channel 0", channels[:1]), (f"all {len(channels)} channels", channels)):
            events = {name: 0 for name in CLASSES}
            samples = 0
            for x in chans:
                tr = stream_scores(model, _to_int16(x * (target / rms(x))))
                for name in CLASSES:
                    events[name] += count_events(tr[name], threshold)
                samples += x.size
            hours = samples / RATE / 3600
            total = sum(events.values())
            lo, hi = poisson_ci(total, hours)
            noise_rows.append({"level": f"{snr} dB", "stream": label, "seconds": samples / RATE,
                               "hours": hours, "events": events, "total": total,
                               "per_hour": total / hours, "per_hour_ci95": [lo, hi]})

    OUT_JSON.write_text(json.dumps({
        "threshold": threshold, "seed": SEED, "noise": "DREGON hovering, eval partition (last 30%)",
        "reference_active_rms": ref_level, "conditions": conditions, "noise_alone": noise_rows,
        "peaks": peaks}, indent=2) + "\n", encoding="utf-8")

    def ms(v: float | None) -> str:
        return "--" if v is None else f"{v:.0f} ms"

    eval_s = channels[0].size / RATE
    lines = [
        "# Branch A under rotor noise (real takes + DREGON)",
        "",
        "Generated by `python eval/wake_noise.py`. Do not edit by hand.",
        "",
        f"The 40 recorded takes of `results/wake_real_voice.md` (one speaker, 20 per class), mixed "
        f"with DREGON hovering ego-noise from the **evaluation partition** (last 30% of the flight, "
        f"{eval_s:.1f} s; training augmentation used the first 70%) by `data/mix_noise.py`'s rule: SNR on "
        f"the active-speech level, one excerpt per take reused at every SNR, seed {SEED}. Noise also "
        f"fills the 1 s pads either side. Scored as `eval/wake_real.py` scores the clean takes, at the "
        f"deployed threshold **{threshold}**; the threshold was not tuned on these mixes. The take is "
        "the unit (n = 20 per class); intervals are exact (Clopper-Pearson).",
        "",
        "## False rejects and class confusions",
        "",
        "| condition | class | n | missed | FRR | 95% CI (exact) | fired own | fired other class | delay p50 | delay p95 |",
        "| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for cond, c in conditions.items():
        for r in c["rows"]:
            lines.append(f"| {cond} | `{r['class']}` | {r['n']} | {r['missed']} | {r['frr']:.3f} "
                         f"| [{r['frr_ci95'][0]:.3f}, {r['frr_ci95'][1]:.3f}] | {r['fired_own']} "
                         f"| {r['fired_other']} | {ms(r['delay_p50_ms'])} | {ms(r['delay_p95_ms'])} |")
    lines += [
        "",
        "## False triggers on rotor noise alone (debounce 1.0 s)",
        "",
        "Noise played at the level it had in the mixes: RMS = median active-speech RMS of the 40 "
        "takes / 10^(SNR/20).",
        "",
        "| level | stream | seconds | hold | abort | total | per hour | 95% CI (Poisson) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for r in noise_rows:
        lines.append(f"| {r['level']} | {r['stream']} | {r['seconds']:.1f} | {r['events']['swarm_hold']} "
                     f"| {r['events']['swarm_abort']} | {r['total']} | {r['per_hour']:.1f} "
                     f"| [{r['per_hour_ci95'][0]:.1f}, {r['per_hour_ci95'][1]:.1f}] |")

    def missed_list(cond: str) -> str:
        return "; ".join(f"`{r['class']}`: " + (", ".join(
            f"{t} ({peaks[cond][t][r['class']]:.4f})" for t in r["missed_takes"]) or "none")
            for r in conditions[cond]["rows"])

    attenuated = sum(v["peak_attenuation"] < 1.0 for c in conditions.values() for v in c["mix"].values())
    lines += [
        "",
        "## Read before quoting",
        "",
        f"- **NFR-16 is FRR <= {NFR16_FRR:.2f}**, judged on the whole interval. At n = 20 even zero "
        "misses leaves an upper bound of ~0.17, so no row here can meet it; a row whose lower bound "
        "exceeds 0.10 misses it.",
        "- The `clean` row is this script's own run with no noise and must equal "
        "`results/wake_real_voice.md`; if it does not, the noisy rows are not comparable to it.",
        "- Missed takes, with the peak their own head reached: " + " | ".join(
            f"{cond}: {missed_list(cond)}" for cond in conditions) + ". Per-take peaks for both heads "
        "are in `results/wake_noise.json`.",
        "- \"Fired other class\" counts takes on which the *other* head reached the threshold "
        "(a take can fire both). A hold heard as abort, or the reverse, is still a stop "
        "(prd.md SS4.2), so a confusion is fail-safe; a miss is not.",
        f"- **The noise-alone streams are short.** The eval partition is {eval_s:.1f} s; with zero "
        "events the Poisson upper bound is several hundred per hour, far above NFR-15's "
        f"{NFR15_FA_PER_HOUR:.0f}/h. They can show that rotor noise by itself does or does not set the "
        "spotter off at a high rate, not that the criterion is met. The eight channels are eight "
        "microphones on one airframe at the same moments, not eight independent recordings.",
        f"- The 40 takes share one {eval_s:.1f} s noise segment, so excerpts overlap between takes; the "
        "variation in noise across takes is narrower than a real flight's.",
        f"- {attenuated} of {sum(len(c['mix']) for c in conditions.values())} mixtures were attenuated "
        "for headroom (SNR unchanged).",
        "- One speaker, one room, one flight recording; digital mixing holds the voice fixed, so "
        "this isolates noise, and it does not capture how the operator raises their voice over rotors "
        "(the Lombard effect).",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {OUT_MD.relative_to(REPO)} and {OUT_JSON.relative_to(REPO)}")
    for cond, c in conditions.items():
        for r in c["rows"]:
            print(f"  {cond:6s} {r['class']:12s} missed {r['missed']}/{r['n']} FRR {r['frr']:.3f} "
                  f"[{r['frr_ci95'][0]:.3f}, {r['frr_ci95'][1]:.3f}] other {r['fired_other']} "
                  f"missed {r['missed_takes']}")
    for r in noise_rows:
        print(f"  noise {r['level']:6s} {r['stream']:14s} {r['seconds']:6.1f} s  events {r['events']}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
