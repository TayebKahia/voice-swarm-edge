#!/usr/bin/env python3
"""Few-shot, schema-in-prompt decoding without fine-tuning: the baseline of A-M2(i).

The article's zero-shot row (EM 0.000 on all 590 items) used the one-line system prompt
of `eval/fixed_audit_prompts.json`, which names a schema it never defines, so it shows
only that the base models do not know the schema. This script gives the same base
models the best fair chance short of training:

- **The schema in the system prompt** (`SYSTEM_PROMPT`): every intent and slot of
  `schema/cmd.gbnf`, units, the 0-based drone ids, the yaw sign, and the rule that
  anything else is `unknown`. Every convention stated is one the training split shows.
- **Ten examples, one per intent**, as chat turns before the item. They come from the
  training split by a fixed rule (`shots()`): clean, not a hard negative, most slots,
  then shortest, then id. No test item is used.
- **The same grammar** (`schema/cmd.gbnf`), greedy decoding, one slot, and the
  tokenisation path of `eval/surface_b.py` (`/apply-template`, then `/tokenize` without
  added specials). One difference: the server's prompt cache is **on**. The schema and
  the ten examples are a prefix of about 800 tokens shared by every item, and
  re-reading it costs about 5 s an item on this workstation, some 2.5 hours a model.
  With one slot and sequential requests the run is still reproducible; what the cache
  changes is only the floating-point order in which the shared prefix was computed.

Weights: the base Instruct checkpoints from their official repositories
(`train/base_models/`, not tracked), converted to f16 GGUF and quantised to Q4_K_M with
the `llama.cpp` build that produced the six fine-tuned artefacts, so the only
difference from the deployed artefacts is the fine-tuning (and the prompt).

Accuracy is host-independent (greedy, fixed grammar), so this runs on the workstation.
Scoring and statistics are `eval/baseline_report.py`'s.

Usage:
    python eval/baseline_fewshot.py                    # run what is missing, then analyse
    python eval/baseline_fewshot.py --analyse-only
    python eval/baseline_fewshot.py --models qwen2.5-0.5b-instruct
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):  # `python eval/baseline_fewshot.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import baseline_report as br
from eval import surface_b

REPO = Path(__file__).resolve().parent.parent
BASE_GGUF = REPO / "train" / "base_models" / "gguf"
PREDS = REPO / "results" / "baseline_fewshot_preds"
OUT_MD = REPO / "results" / "baseline_fewshot.md"
GRAMMAR = (REPO / "schema" / "cmd.gbnf").read_text()
PORT = 8918  # clear of 8912-8917
QUANT = "Q4_K_M"

MODELS = ("qwen2.5-0.5b-instruct", "llama-3.2-1b-instruct", "smollm2-360m-instruct")
SHORT = {"qwen2.5-0.5b-instruct": "Qwen2.5-0.5B", "llama-3.2-1b-instruct": "Llama-3.2-1B",
         "smollm2-360m-instruct": "SmolLM2-360M"}
#: Where each base artefact came from. The weights are not tracked, so this is the record:
#: the Hugging Face revision downloaded, and the SHA-256 of the Q4_K_M file that was
#: decoded. Conversion: `convert_hf_to_gguf.py --outtype f16` and `llama-quantize ... Q4_K_M`
#: at `llama.cpp` 88ada91c1, in a throwaway environment built from that commit's
#: `requirements-convert_hf_to_gguf.txt` (torch 2.11.0+cpu, transformers 4.57.6).
PROVENANCE = {
    "qwen2.5-0.5b-instruct": ("Qwen/Qwen2.5-0.5B-Instruct",
                              "7ae557604adf67be50417f59c2c2f167def9a775",
                              "d1dbc29d545ed1b6cd9af89e1ef11b9361fe8cce68aa9e36de513d710e486f6a"),
    "llama-3.2-1b-instruct": ("meta-llama/Llama-3.2-1B-Instruct",
                              "9213176726f574b556790deb65791e0c5aa438b6",
                              "4b67d7dfac0c5926d13be3dc5f94b901f7fe929b1206bf337dc3169cee74d47d"),
    "smollm2-360m-instruct": ("HuggingFaceTB/SmolLM2-360M-Instruct",
                              "a10cc1512eabd3dde888204e902eca88bddb4951",
                              "4a30f27a607fb65cba9f82f451a2e99b1bd233954f6d91c95e8853a634668126"),
}
INTENTS = ("formation", "move", "altitude", "takeoff", "land", "hover", "abort", "rotate",
           "set_param", "unknown")

SYSTEM_PROMPT = """\
You are a drone swarm command parser. Turn the operator's utterance into exactly one \
JSON command and output nothing else. Omit every field the utterance does not give.

Commands (numbers in metres, metres per second or degrees, at most one decimal):
- {"intent":"formation","shape":S,"radius":r,"spacing":s,"ids":[...]} with S one of \
"circle","line","wedge","grid","column","flock"; radius only for a circle (half of a \
stated diameter), spacing is the distance between drones.
- {"intent":"move","dir":D,"dist":d,"speed":v,"ids":[...]} with D one of \
"north","south","east","west","up","down","forward","back","left","right"; or \
{"intent":"move","pos":[x,y,z],"speed":v,"ids":[...]} for a target position.
- {"intent":"altitude","z":z,"ids":[...]}: go to a height.
- {"intent":"takeoff","z":z,"ids":[...]}: take off, optionally to a height.
- {"intent":"land","ids":[...]}
- {"intent":"hover","ids":[...]}: hold position, stop what you are doing, or cancel.
- {"intent":"abort"}: emergency stop, kill the motors.
- {"intent":"rotate","yaw":a,"ids":[...]}: turn by a degrees; left or \
counter-clockwise is positive, right or clockwise is negative.
- {"intent":"set_param","speed":v,"spacing":s,"alt":h}: change the default cruise \
speed, spacing or cruise altitude.
- {"intent":"unknown"}: anything that is not one of these commands.

Drones are numbered 1 to 5 when spoken and 0 to 4 in "ids": "drone 1" is [0], \
"units 3 to 5" is [2,3,4]. Leave "ids" out when the whole swarm is addressed."""


def shots() -> list[dict]:
    """One training example per intent: clean, not a hard negative, most slots, shortest."""
    rows = br.split_items("train")
    out = []
    for intent in INTENTS:
        pool = [r for r in rows if r["gold_intent"] == intent and r["variant"] == "clean"
                and r["source_variant"] != "hard_negative"]
        pool.sort(key=lambda r: (-len(json.loads(r["target"])), len(r["transcript"]), r["id"]))
        out.append(pool[0])
    return out


def messages(shot_rows: Sequence[dict], text: str) -> list[dict]:
    turns = [{"role": "system", "content": SYSTEM_PROMPT}]
    for row in shot_rows:
        turns += [{"role": "user", "content": row["transcript"]},
                  {"role": "assistant", "content": row["target"]}]
    return turns + [{"role": "user", "content": text}]


def predict(shot_rows: Sequence[dict], text: str) -> str:
    """`eval/surface_b.py`'s decode path, with the few-shot turns in the template."""
    templated = surface_b._post("/apply-template", {"messages": messages(shot_rows, text)})["prompt"]
    tokens = surface_b._post("/tokenize", {"content": templated, "add_special": False})["tokens"]
    return surface_b._post("/completion", {
        "prompt": tokens, "temperature": 0.0, "n_predict": surface_b.MAX_TOKENS,
        "cache_prompt": True, "grammar": GRAMMAR})["content"].strip()


def run_model(model: str, fresh: bool) -> None:
    out_dir = PREDS / model
    pending_splits = [s for s in br.SPLITS if fresh or not (out_dir / f"{s}.jsonl").is_file()]
    pending_asr = fresh or not (out_dir / "asr" / "5.jsonl").is_file()
    if not pending_splits and not pending_asr:
        print(f"{model}: complete")
        return
    gguf = BASE_GGUF / f"{model}-base-{QUANT}.gguf"
    if not gguf.is_file():
        print(f"{model}: no base artefact at {gguf.relative_to(REPO)}, skipped")
        return

    surface_b.PORT = PORT  # surface_b's HTTP helpers read the module-level port
    shot_rows = shots()
    context = 4096  # the ten examples and the schema do not fit surface_b's 1024
    process = subprocess.Popen(
        [str(surface_b.LLAMA_SERVER), "-m", str(gguf), "--port", str(PORT),
         "-t", str(surface_b.THREADS), "-c", str(context), "--parallel", "1", "--log-disable"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        surface_b._wait_for_server(process)
        for split in pending_splits:
            started, rows = time.monotonic(), []
            for row in br.split_items(split):
                rows.append({"id": row["id"], "transcript": row["transcript"],
                             "gold": row["target"], "raw": predict(shot_rows, row["transcript"])})
            br.write_jsonl(out_dir / f"{split}.jsonl", rows)
            print(f"  {model:24s} {split:12s} {len(rows)} items "
                  f"({len(rows) / (time.monotonic() - started):.1f}/s)", flush=True)
        if pending_asr:
            br.run_asr(lambda text: predict(shot_rows, text), out_dir / "asr")
            print(f"  {model:24s} ASR transcripts done", flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()


def analyse() -> int:
    br.quiet()
    done = [m for m in MODELS if (PREDS / m / "asr" / "5.jsonl").is_file()]
    if not done:
        raise SystemExit("no few-shot predictions yet")
    systems, asr = {}, {}
    for model in done:
        name = f"{SHORT[model]} few-shot"
        systems[name] = {s: br.predictions_from_rows(br.read_jsonl(PREDS / model / f"{s}.jsonl"))
                         for s in br.SPLITS}
        asr[name] = PREDS / model / "asr"
    for stem in br.FINE_TUNED:
        systems[br.SHORT[stem]] = {s: br.fine_tuned_predictions(stem, s) for s in br.SPLITS}
        asr[br.SHORT[stem]] = br.ASR_ARTEFACTS / stem
    reference = br.SHORT[br.DEPLOYED]

    shot_ids = ", ".join(f"`{r['id']}`" for r in shots())
    missing = [m for m in MODELS if m not in done]
    lines = [
        "### Few-shot, schema-in-prompt baseline against the fine-tuned models (A-M2 i)",
        "",
        "Generated by `python eval/baseline_fewshot.py`. Do not edit by hand.",
        "",
        f"Base Instruct models, no fine-tuning, quantised to {QUANT} by the `llama.cpp` "
        "build of the fine-tuned artefacts; decoded greedily under `schema/cmd.gbnf`. "
        "Prompt: the schema in the system prompt (`SYSTEM_PROMPT` in the script) and ten "
        f"training examples, one per intent, as chat turns ({shot_ids}). Fine-tuned rows "
        "are the same three models after fine-tuning, at the same quantisation and under "
        "the same grammar, with their own one-line prompt. Raw predictions: "
        "`results/baseline_fewshot_preds/`.",
        *([f"", f"Not run: {', '.join(missing)} (no base artefact)."] if missing else []),
        "",
        "| Base model | Hugging Face revision | Q4_K_M SHA-256 |",
        "| :--- | :--- | :--- |",
        *[f"| `{PROVENANCE[m][0]}` | `{PROVENANCE[m][1][:12]}` | `{PROVENANCE[m][2][:16]}...` |"
          for m in done],
        "",
        "Conversion: `convert_hf_to_gguf.py --outtype f16`, then `llama-quantize` to Q4_K_M, "
        "both at `llama.cpp` 88ada91c1 (torch 2.11.0+cpu, transformers 4.57.6). The prompt "
        "cache is on for this baseline (shared 800-token prefix); see the script's docstring.",
        "",
        *br.text_tables(systems, reference),
        "",
        *br.asr_tables(asr, reference),
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    args = parser.parse_args(argv)
    br.quiet()
    if not args.analyse_only:
        for model in args.models:
            run_model(model, args.fresh)
    return analyse()


if __name__ == "__main__":
    sys.exit(main())
