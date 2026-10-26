### Golden-set errors by template family

Per-item predictions, scored by `eval/metrics.py`. Deployed rows: `llama.cpp`,
grammar on. fp16 rows: the fine-tuned reference model under `transformers`.
200 items in 12 template families.

#### Exact match with item-level and family-clustered intervals

Item-level: items resampled (as `results/em_ci.md`). Clustered: the 12 families
resampled whole. Both percentile bootstraps, 10,000 replicates, fixed seed.
`without F106`: the 178 items outside the rotate family.

| Configuration | EM | 95% CI, items | 95% CI, families | EM without F106 |
| :--- | ---: | ---: | ---: | ---: |
| Llama Q4_K_M | 0.910 | [0.865, 0.945] | [0.794, 0.985] | 0.966 |
| Llama Q8_0 | 0.915 | [0.875, 0.950] | [0.799, 0.989] | 0.972 |
| Qwen Q4_K_M | 0.935 | [0.900, 0.965] | [0.828, 1.000] | 0.989 |
| Qwen Q8_0 | 0.930 | [0.890, 0.965] | [0.823, 1.000] | 0.983 |
| SmolLM2 Q4_K_M | 0.760 | [0.700, 0.815] | [0.649, 0.867] | 0.803 |
| SmolLM2 Q8_0 | 0.790 | [0.730, 0.845] | [0.659, 0.920] | 0.837 |
| Llama fp16 | 0.925 | [0.885, 0.960] | [0.818, 0.994] | 0.978 |
| Qwen fp16 | 0.935 | [0.900, 0.965] | [0.828, 1.000] | 0.989 |
| SmolLM2 fp16 | 0.775 | [0.715, 0.830] | [0.636, 0.914] | 0.826 |
| Danube3 fp16 | 0.870 | [0.820, 0.915] | [0.736, 0.963] | 0.933 |

#### Correct items per family

| Family | Intent | n | Llama Q4_K_M | Qwen Q4_K_M | SmolLM2 Q4_K_M | Qwen fp16 | Danube3 fp16 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| F005 | formation | 17 | 16 | 16 | 14 | 16 | 15 |
| F010 | formation | 17 | 15 | 17 | 11 | 17 | 15 |
| F030 | move | 17 | 15 | 16 | 9 | 16 | 14 |
| F043 | move | 17 | 17 | 17 | 16 | 17 | 17 |
| F066 | altitude | 22 | 21 | 22 | 17 | 22 | 22 |
| F076 | takeoff | 16 | 16 | 16 | 11 | 16 | 13 |
| F084 | land | 12 | 12 | 12 | 12 | 12 | 11 |
| F092 | hover | 12 | 12 | 12 | 10 | 12 | 12 |
| F098 | abort | 10 | 10 | 10 | 10 | 10 | 10 |
| F106 | rotate | 22 | 10 | 11 | 9 | 11 | 8 |
| F114 | set_param | 22 | 22 | 22 | 17 | 22 | 22 |
| F120 | unknown | 16 | 16 | 16 | 16 | 16 | 15 |
| all | | 200 | 182 | 187 | 152 | 187 | 174 |

#### Paired differences in exact match

A minus B on the same items. b / c: items only A / only B gets right; p: exact
McNemar (item-level). Clustered interval: families resampled whole.

| A | B | A - B (pp) | b / c | exact p | 95% CI, families (pp) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Qwen Q4_K_M | Llama Q4_K_M | +2.5 | 5 / 0 | 0.0625 | [+0.5, +4.7] |
| Qwen Q4_K_M | SmolLM2 Q4_K_M | +17.5 | 38 / 3 | 1.05e-08 | [+10.0, +25.0] |
| Llama Q4_K_M | SmolLM2 Q4_K_M | +15.0 | 34 / 4 | 6.04e-07 | [+8.4, +21.5] |
| Qwen fp16 | Danube3 fp16 | +6.5 | 15 / 2 | 0.00235 | [+2.9, +10.3] |
| Llama Q4_K_M | Llama Q8_0 | -0.5 | 0 / 1 | 1 | [-1.6, +0.0] |
| Qwen Q4_K_M | Qwen Q8_0 | +0.5 | 1 / 0 | 1 | [+0.0, +1.6] |
| SmolLM2 Q4_K_M | SmolLM2 Q8_0 | -3.0 | 8 / 14 | 0.286 | [-10.1, +5.3] |

#### Qwen against Llama at Q4_K_M on every split

| Split | n | EM Qwen | EM Llama | only Qwen / only Llama | exact p |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `test_synth` | 240 | 0.954 | 0.958 | 3 / 4 | 1 |
| `test_golden` | 200 | 0.935 | 0.910 | 5 / 0 | 0.0625 |
| `test_ood` | 150 | 0.747 | 0.813 | 7 / 17 | 0.0639 |

#### Direction phrasing of rotate commands

Cue: the first match, in order, of an explicit minus, counter-clockwise,
clockwise, left, right. The schema encodes counter-clockwise as a positive yaw.

Training (`data/train.jsonl`, 120 rotate pairs):

| Cue | yaw > 0 | yaw < 0 |
| :--- | ---: | ---: |
| minus | 0 | 11 |
| counter-clockwise | 20 | 0 |
| clockwise | 0 | 6 |
| left | 43 | 0 |
| right | 1 | 20 |
| none | 17 | 2 |

Golden rotate items, and errors at Q4_K_M:

| Cue | items | yaw < 0 | Llama wrong | Qwen wrong | SmolLM2 wrong |
| :--- | ---: | ---: | ---: | ---: | ---: |
| minus | 2 | 2 | 0 | 0 | 0 |
| counter-clockwise | 1 | 0 | 0 | 0 | 0 |
| clockwise | 5 | 5 | 5 | 5 | 4 |
| left | 4 | 0 | 0 | 0 | 2 |
| right | 7 | 7 | 7 | 6 | 6 |
| none | 3 | 0 | 0 | 0 | 1 |
