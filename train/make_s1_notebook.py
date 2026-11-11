#!/usr/bin/env python3
"""Generate train/kaggle_s1_rotation.ipynb, the S1 retraining notebook (review A-M4b).

    python train/make_s1_notebook.py

The notebook is `train/kaggle_finetune.ipynb` with a handful of explicit, asserted
patches, so every line of the deployed training path (prompt, masking, LoRA,
Trainer arguments, best-val-EM checkpoint, Surface-A evaluation, parity dump,
merge and GGUF conversion) is literally the same code. Only these change:

* the header cell, which says what this session is;
* the run list. Instead of one config per model there are four runs:

      qwen2.5-0.5b-instruct-s1      train_s1_balanced.jsonl  (the fix)
      llama-3.2-1b-instruct-s1      train_s1_balanced.jsonl  (the fix)
      qwen2.5-0.5b-instruct-ctrl    train.jsonl              (rerun control)
      llama-3.2-1b-instruct-ctrl    train.jsonl              (rerun control)

  The control reruns the deployed data in this same session. fp16 GPU training is
  not bit-reproducible, so "fix minus deployed" mixes the data change with rerun
  noise, and "fix minus control" isolates the data change;
* each run gets the deployed config unchanged (same hyperparameters, same seed 42),
  a name suffix, and its own training rows, and the RNGs are reset to seed 42
  before each run so that the LoRA initialisation does not depend on run order;
* the final instructions, which point at `train/kaggle_out_s1/`.

Nothing here touches the deployed notebook, adapters or GGUFs.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SOURCE = HERE / "kaggle_finetune.ipynb"
OUTPUT = HERE / "kaggle_s1_rotation.ipynb"

HEADER = r"""# PFE --- S1 retrain: rotation-balanced data (review A-M4b)

**This session does not replace the deployed models.** It trains four LoRA runs whose
outputs go to `train/kaggle_out_s1/` and are reported in the master article as a
retrained variant. The deployed adapters and GGUFs stay as they are.

| run | config (unchanged) | training file |
| :--- | :--- | :--- |
| `qwen2.5-0.5b-instruct-s1` | `configs/qwen2.5-0.5b.yaml` | `data/train_s1_balanced.jsonl` |
| `llama-3.2-1b-instruct-s1` | `configs/llama-3.2-1b.yaml` | `data/train_s1_balanced.jsonl` |
| `qwen2.5-0.5b-instruct-ctrl` | `configs/qwen2.5-0.5b.yaml` | `data/train.jsonl` (deployed data) |
| `llama-3.2-1b-instruct-ctrl` | `configs/llama-3.2-1b.yaml` | `data/train.jsonl` (deployed data) |

`train_s1_balanced.jsonl` is `train.jsonl` with 19 rotation rows mirrored (yaw negated,
left/counter-clockwise swapped for right/clockwise; `data/s1_balance_rotation.py`).
Every other row is byte-identical. The control reruns the deployed data in this same
session, because fp16 GPU training is not bit-reproducible.

Expect roughly **2--3 hours** (two Qwen and two Llama runs, plus evaluation and GGUF
conversion).

## Before you press Run All

1. **Accelerator: GPU T4 x2** (Settings -> Accelerator), as for the deployed runs.
2. **Internet: ON** (Settings -> Internet).
3. **Dataset attached:** the zip from `python train/make_kaggle_bundle.py --s1`
   (`train/pfe_kaggle_data_s1.zip`). It holds both training files.
4. **`HF_TOKEN` secret attached** (Add-ons -> Secrets). Llama-3.2-1B is gated. Without
   the token the two Llama runs are skipped and the notebook says so.
5. **Download before the session expires:** `/kaggle/working/pfe_outputs.zip`, and the
   four `gguf/*-f16.gguf` files one by one from the Output panel.
"""

WARMUP_SHIM = r'''import inspect as _inspect
import transformers as _transformers

def _warmup(ratio):
    """warmup_ratio on transformers 4; on 5, where it was removed, the same ratio as warmup_steps."""
    if "warmup_ratio" in _inspect.signature(TrainingArguments.__init__).parameters:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": float(ratio)}

print("transformers", _transformers.__version__, "| warmup kwargs:", _warmup(0.03))

'''

RUNS_CELL = r'''
# S1: four runs, each the deployed config unchanged plus a name suffix and its own
# training rows. See the header cell.
RUNS = [
    ("configs/qwen2.5-0.5b.yaml", "-s1", "train_s1_balanced"),
    ("configs/llama-3.2-1b.yaml", "-s1", "train_s1_balanced"),
    ("configs/qwen2.5-0.5b.yaml", "-ctrl", "train"),
    ("configs/llama-3.2-1b.yaml", "-ctrl", "train"),
]
TRAIN_FILES = {stem: load(stem) for stem in {stem for _, _, stem in RUNS}}
print({stem: len(rows) for stem, rows in TRAIN_FILES.items()})
assert len(TRAIN_FILES["train"]) == len(TRAIN_FILES["train_s1_balanced"])
_diff = sum(a["target"] != b["target"]
            for a, b in zip(TRAIN_FILES["train"], TRAIN_FILES["train_s1_balanced"]))
print(f"rows whose target differs between the two training files: {_diff}")
assert _diff == 19, _diff

trained, failed, started = [], [], time.time()
for relative, RUN_SUFFIX, stem in RUNS:
    TRAIN = TRAIN_FILES[stem]
    # Same starting RNG state for every run, so the LoRA initialisation and data
    # order do not depend on which runs came before it in this session.
    set_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    print(f"\n>>> run suffix {RUN_SUFFIX}  training file {stem}.jsonl  ({len(TRAIN)} rows)")
    try:
        entry = train_one(DATA / relative)
    except Exception as exc:
        import traceback
        failed.append((relative + RUN_SUFFIX, f"{type(exc).__name__}: {exc}"))
        print(f"\n!! {relative}{RUN_SUFFIX} FAILED -- continuing with the rest")
        traceback.print_exc()
        torch.cuda.empty_cache()
        continue
    if entry:
        entry["train_file"] = stem
        trained.append(entry)
        entry["model"].cpu(); torch.cuda.empty_cache()

print(f"\ntrained {len(trained)}/{len(RUNS)} in {(time.time() - started) / 60:.1f} min")
for relative, reason in failed:
    print(f"  FAILED {relative}: {reason}")
if not trained:
    raise SystemExit("no model trained -- nothing downstream can run")
'''

FINAL_NOTE = '''DOWNLOAD pfe_outputs.zip NOW -- nothing here survives the session.
Also download the four gguf/*-f16.gguf files one by one from the Output panel.

On the workstation (S1 outputs stay apart from the deployed ones):

  mkdir -p train/kaggle_out_s1/gguf
  unzip -o pfe_outputs.zip -d train/kaggle_out_s1/
  mv ~/Downloads/*-s1-f16.gguf ~/Downloads/*-ctrl-f16.gguf train/kaggle_out_s1/gguf/
  python train/quantise.py --input train/kaggle_out_s1/gguf --output gguf_s1
'''


def _replace_once(text: str, old: str, new: str, where: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{where}: expected exactly one {old[:60]!r}, found {count}")
    return text.replace(old, new)


def build() -> dict:
    nb = json.loads(SOURCE.read_text())
    cells = nb["cells"]
    sources = ["".join(c["source"]) for c in cells]

    def find(marker: str) -> int:
        hits = [i for i, s in enumerate(sources) if marker in s]
        if len(hits) != 1:
            raise SystemExit(f"expected one cell containing {marker!r}, found {hits}")
        return hits[0]

    def put(i: int, text: str) -> None:
        sources[i] = text
        cells[i]["source"] = text.strip("\n").splitlines(keepends=True)

    put(0, HEADER)

    i = find('name, hf_id = cfg["name"], cfg["hf_id"]')
    text = _replace_once(sources[i], 'name, hf_id = cfg["name"], cfg["hf_id"]',
                         'name, hf_id = cfg["name"] + RUN_SUFFIX, cfg["hf_id"]', "train_one")
    # Kaggle's image moved to transformers 5 after the deployed runs, and 5 removed
    # `warmup_ratio`: a float below 1 passed as `warmup_steps` is the same ratio
    # (ceil(ratio x total steps)). The value stays 0.03 from the config either way.
    text = _replace_once(text, 'warmup_ratio=cfg["optim"]["warmup_ratio"],',
                         '**_warmup(cfg["optim"]["warmup_ratio"]),', "train_one")
    text = WARMUP_SHIM + text
    put(i, text)

    put(find("ONLY: list[str] ="), RUNS_CELL)

    i = find("DOWNLOAD pfe_outputs.zip NOW")
    head, _, _ = sources[i].partition('print("""')
    put(i, head + 'print("""\n' + FINAL_NOTE + '""")\n')

    # The adapter tensors are loaded per run by name, so the suffixed names flow
    # through Surface-A, the parity dump and the merge/convert cells unchanged.
    return nb


def main() -> int:
    nb = build()
    OUTPUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUTPUT.relative_to(REPO)}  ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
