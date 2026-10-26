# Data licence

The code in this repository is licensed under the MIT licence (`LICENSE`). The data the
authors created are licensed under the **Creative Commons Attribution 4.0 International**
licence (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

## Covered by CC BY 4.0

- The command grammar (`schema/cmd.gbnf`); the schema code in `schema/` is MIT.
- The generated command corpus and its splits (`data/*.jsonl`), including the
  out-of-domain set.
- The golden-set and keyword recordings (`data/audio/s1/`, `data/audio/s1_16k/`,
  `data/audio/wake_real/`, `data/audio/wake_real_16k/`). They are recordings of the
  first author, who consents to their release under this licence.
- Measurement records, predictions and analysis outputs (`results/`).

Attribution: Tayeb Kahia and Belkacem Khaldi, École Supérieure en Informatique –
08 Mai 1945, Sidi Bel Abbès, Algeria.

## Not covered: third-party material keeps its own terms

- **Noise and speech corpora** (DREGON, ESC-50, LibriSpeech, Speech Commands, Common
  Voice) are not redistributed here. Noise-mixed audio is regenerated from the original
  corpora by `data/mix_noise.py`, under each corpus's own licence.
- **Model weights.** Base models and any merged or quantised artefacts keep their
  model licences; see `gguf/LICENSE_MODELS.md`. Artefacts derived from
  Llama-3.2-1B-Instruct are subject to the Meta Llama 3.2 Community Licence.
- **Synthesised speech** produced with Piper voices is subject to each voice's licence.
