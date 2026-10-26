### Out-of-domain set by source

`data/test_ood.jsonl`, 150 items, every reference `unknown`.

| Source | Provenance | Items |
| :--- | :--- | ---: |
| asr_garbage | authored_asr_garbage_pending_harvest | 25 |
| asr_garbage | whisper_tiny.en_harvested | 12 |
| drone_adjacent_unsupported | authored | 37 |
| massive_assistant | massive_taxonomy_authored | 38 |
| truncated_fragment | authored | 38 |

False-command rate (dispatched action other than `unknown`) per source,
grammar on:

| Configuration | asr_garbage | drone_adjacent_unsupported | massive_assistant | truncated_fragment | all |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Llama Q4_K_M | 1/37 | 8/37 | 0/38 | 19/38 | 0.187 |
| Llama Q8_0 | 1/37 | 9/37 | 0/38 | 20/38 | 0.200 |
| Qwen Q4_K_M | 3/37 | 12/37 | 3/38 | 20/38 | 0.253 |
| Qwen Q8_0 | 2/37 | 13/37 | 3/38 | 22/38 | 0.267 |
| SmolLM2 Q4_K_M | 12/37 | 17/37 | 2/38 | 20/38 | 0.340 |
| SmolLM2 Q8_0 | 14/37 | 20/37 | 3/38 | 22/38 | 0.393 |

### Grammar ablation per artefact

The 590 items of the three test splits, greedy, grammar on and off; nothing else
changes. Malformed: output that does not parse as a schema-valid command. Upper
bound: exact one-sided 95% (Clopper-Pearson) bound on the malformed rate without
the grammar. b / c: items only grammar-on / only grammar-off gets right.

| Artefact | Malformed on | Malformed off | 95% upper bound, off | EM on | EM off | b / c | exact p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama Q4_K_M | 0 | 1 | 0.0080 | 0.9051 | 0.9034 | 1 / 0 | 1 |
| Llama Q8_0 | 0 | 0 | 0.0051 | 0.9051 | 0.9051 | 0 / 0 | 1 |
| Qwen Q4_K_M | 0 | 0 | 0.0051 | 0.8949 | 0.8915 | 2 / 0 | 0.5 |
| Qwen Q8_0 | 0 | 0 | 0.0051 | 0.8915 | 0.8881 | 2 / 0 | 0.5 |
| SmolLM2 Q4_K_M | 0 | 0 | 0.0051 | 0.7559 | 0.7542 | 1 / 0 | 1 |
| SmolLM2 Q8_0 | 0 | 0 | 0.0051 | 0.7525 | 0.7525 | 0 / 0 | 1 |
