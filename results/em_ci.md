### Exact match with 95% bootstrap intervals

Re-derived from the per-item predictions (`results/surface_b_preds/`, grammar on;
`train/kaggle_out/preds_*.jsonl` for the fp16 reference) by `eval/em_ci.py`.
Percentile bootstrap over items, 10,000 replicates, fixed seed,
nearest-rank bounds. The golden items are paraphrases of 12 command patterns, so
the item-level interval is narrower than the true uncertainty.

| Surface | Model | Quant | Split | n | EM | 95% CI |
| :--- | :--- | :--- | :--- | ---: | ---: | :--- |
| deployed | llama-3.2-1b-instruct | Q4_K_M | `test_synth` | 240 | 0.958 | [0.933, 0.983] |
| deployed | llama-3.2-1b-instruct | Q4_K_M | `test_golden` | 200 | 0.910 | [0.865, 0.945] |
| deployed | llama-3.2-1b-instruct | Q4_K_M | `test_ood` | 150 | 0.813 | [0.747, 0.873] |
| deployed | llama-3.2-1b-instruct | Q8_0 | `test_synth` | 240 | 0.963 | [0.938, 0.983] |
| deployed | llama-3.2-1b-instruct | Q8_0 | `test_golden` | 200 | 0.915 | [0.875, 0.950] |
| deployed | llama-3.2-1b-instruct | Q8_0 | `test_ood` | 150 | 0.800 | [0.733, 0.860] |
| deployed | qwen2.5-0.5b-instruct | Q4_K_M | `test_synth` | 240 | 0.954 | [0.925, 0.979] |
| deployed | qwen2.5-0.5b-instruct | Q4_K_M | `test_golden` | 200 | 0.935 | [0.900, 0.965] |
| deployed | qwen2.5-0.5b-instruct | Q4_K_M | `test_ood` | 150 | 0.747 | [0.673, 0.813] |
| deployed | qwen2.5-0.5b-instruct | Q8_0 | `test_synth` | 240 | 0.958 | [0.933, 0.983] |
| deployed | qwen2.5-0.5b-instruct | Q8_0 | `test_golden` | 200 | 0.930 | [0.890, 0.965] |
| deployed | qwen2.5-0.5b-instruct | Q8_0 | `test_ood` | 150 | 0.733 | [0.660, 0.800] |
| deployed | smollm2-360m-instruct | Q4_K_M | `test_synth` | 240 | 0.812 | [0.762, 0.858] |
| deployed | smollm2-360m-instruct | Q4_K_M | `test_golden` | 200 | 0.760 | [0.700, 0.815] |
| deployed | smollm2-360m-instruct | Q4_K_M | `test_ood` | 150 | 0.660 | [0.587, 0.733] |
| deployed | smollm2-360m-instruct | Q8_0 | `test_synth` | 240 | 0.812 | [0.762, 0.863] |
| deployed | smollm2-360m-instruct | Q8_0 | `test_golden` | 200 | 0.790 | [0.730, 0.845] |
| deployed | smollm2-360m-instruct | Q8_0 | `test_ood` | 150 | 0.607 | [0.527, 0.687] |
| reference | llama-3.2-1b-instruct | fp16 | `test_synth` | 240 | 0.958 | [0.929, 0.983] |
| reference | llama-3.2-1b-instruct | fp16 | `test_golden` | 200 | 0.925 | [0.885, 0.960] |
| reference | llama-3.2-1b-instruct | fp16 | `test_ood` | 150 | 0.733 | [0.660, 0.800] |
| reference | qwen2.5-0.5b-instruct | fp16 | `test_synth` | 240 | 0.954 | [0.925, 0.979] |
| reference | qwen2.5-0.5b-instruct | fp16 | `test_golden` | 200 | 0.935 | [0.900, 0.965] |
| reference | qwen2.5-0.5b-instruct | fp16 | `test_ood` | 150 | 0.727 | [0.653, 0.793] |
| reference | smollm2-360m-instruct | fp16 | `test_synth` | 240 | 0.825 | [0.775, 0.871] |
| reference | smollm2-360m-instruct | fp16 | `test_golden` | 200 | 0.775 | [0.715, 0.830] |
| reference | smollm2-360m-instruct | fp16 | `test_ood` | 150 | 0.620 | [0.540, 0.693] |
| reference | h2o-danube3-500m-chat | fp16 | `test_synth` | 240 | 0.887 | [0.846, 0.925] |
| reference | h2o-danube3-500m-chat | fp16 | `test_golden` | 200 | 0.870 | [0.820, 0.915] |
| reference | h2o-danube3-500m-chat | fp16 | `test_ood` | 150 | 0.720 | [0.647, 0.787] |
