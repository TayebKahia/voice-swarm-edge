### S1: rotation-balanced retrain (review A-M4b)

`deployed`: the artefacts of record. `ctrl`: retrained on the deployed data in the
S1 session. `s1`: retrained in the same session on `data/train_s1_balanced.jsonl`
(19 rotation rows mirrored; `data/s1_balance_rotation.py`). Same configs and seed.
All decoded by `llama.cpp` under the grammar.

#### Rotation pairs in training, by direction cue

| Cue | deployed yaw > 0 | deployed yaw < 0 | S1 yaw > 0 | S1 yaw < 0 |
| :--- | ---: | ---: | ---: | ---: |
| minus | 0 | 11 | 0 | 13 |
| counter-clockwise | 20 | 0 | 15 | 0 |
| clockwise | 0 | 6 | 0 | 11 |
| left | 43 | 0 | 31 | 0 |
| right | 1 | 20 | 1 | 32 |
| none | 17 | 2 | 15 | 2 |

#### Exact match

95% percentile bootstrap, 10,000 replicates; template families resampled whole on
`test_golden` (12) and `test_synth` (12), items on `test_ood` (one family).

| Model | Quant | Variant | `test_golden` EM [95% CI] | `test_synth` EM [95% CI] | `test_ood` EM [95% CI] |
| :--- | :--- | :--- | ---: | ---: | ---: |
| Qwen | Q4_K_M | deployed | 0.935 [0.828, 1.000] | 0.954 [0.883, 1.000] | 0.747 [0.673, 0.813] (items) |
| Qwen | Q4_K_M | ctrl | 0.935 [0.828, 1.000] | 0.954 [0.883, 1.000] | 0.747 [0.673, 0.813] (items) |
| Qwen | Q4_K_M | s1 | 0.925 [0.826, 1.000] | 0.971 [0.921, 1.000] | 0.767 [0.693, 0.833] (items) |
| Qwen | Q8_0 | deployed | 0.930 [0.823, 1.000] | 0.958 [0.892, 1.000] | 0.733 [0.660, 0.800] (items) |
| Qwen | Q8_0 | ctrl | 0.930 [0.823, 1.000] | 0.958 [0.892, 1.000] | 0.733 [0.660, 0.800] (items) |
| Qwen | Q8_0 | s1 | 0.965 [0.923, 1.000] | 0.971 [0.925, 1.000] | 0.740 [0.667, 0.807] (items) |
| Llama | Q4_K_M | deployed | 0.910 [0.798, 0.985] | 0.958 [0.879, 1.000] | 0.813 [0.747, 0.873] (items) |
| Llama | Q4_K_M | ctrl | 0.915 [0.802, 0.989] | 0.954 [0.875, 1.000] | 0.807 [0.740, 0.867] (items) |
| Llama | Q4_K_M | s1 | 0.915 [0.821, 0.983] | 0.967 [0.917, 1.000] | 0.800 [0.733, 0.860] (items) |
| Llama | Q8_0 | deployed | 0.915 [0.802, 0.989] | 0.963 [0.892, 1.000] | 0.800 [0.733, 0.860] (items) |
| Llama | Q8_0 | ctrl | 0.920 [0.807, 0.994] | 0.958 [0.887, 1.000] | 0.747 [0.673, 0.813] (items) |
| Llama | Q8_0 | s1 | 0.930 [0.844, 0.990] | 0.971 [0.921, 1.000] | 0.760 [0.693, 0.827] (items) |

#### Paired comparisons within each model

A minus B on the same items. b / c: items only A / only B gets right. p: exact
McNemar, item level (items of one family are not independent; read with the
clustered interval).

| Model | Quant | Split | A | B | reads as | A - B (pp) | b / c | exact p | 95% CI (pp) |
| :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| Qwen | Q4_K_M | `test_golden` | s1 | ctrl | the data change | -1.0 | 1 / 3 | 0.625 | [-3.6, +1.0] |
| Qwen | Q4_K_M | `test_golden` | s1 | deployed | against the artefacts of record | -1.0 | 1 / 3 | 0.625 | [-3.6, +1.0] |
| Qwen | Q4_K_M | `test_golden` | ctrl | deployed | rerun noise | +0.0 | 0 / 0 | 1 | [+0.0, +0.0] |
| Qwen | Q4_K_M | `test_synth` | s1 | ctrl | the data change | +1.7 | 4 / 0 | 0.125 | [+0.0, +4.2] |
| Qwen | Q4_K_M | `test_synth` | s1 | deployed | against the artefacts of record | +1.7 | 4 / 0 | 0.125 | [+0.0, +4.2] |
| Qwen | Q4_K_M | `test_synth` | ctrl | deployed | rerun noise | +0.0 | 0 / 0 | 1 | [+0.0, +0.0] |
| Qwen | Q4_K_M | `test_ood` | s1 | ctrl | the data change | +2.0 | 4 / 1 | 0.375 | [-0.7, +5.3] (items) |
| Qwen | Q4_K_M | `test_ood` | s1 | deployed | against the artefacts of record | +2.0 | 4 / 1 | 0.375 | [-0.7, +5.3] (items) |
| Qwen | Q4_K_M | `test_ood` | ctrl | deployed | rerun noise | +0.0 | 0 / 0 | 1 | [+0.0, +0.0] (items) |
| Qwen | Q8_0 | `test_golden` | s1 | ctrl | the data change | +3.5 | 9 / 2 | 0.0654 | [-1.1, +10.6] |
| Qwen | Q8_0 | `test_golden` | s1 | deployed | against the artefacts of record | +3.5 | 9 / 2 | 0.0654 | [-1.1, +10.6] |
| Qwen | Q8_0 | `test_golden` | ctrl | deployed | rerun noise | +0.0 | 0 / 0 | 1 | [+0.0, +0.0] |
| Qwen | Q8_0 | `test_synth` | s1 | ctrl | the data change | +1.2 | 3 / 0 | 0.25 | [+0.0, +3.8] |
| Qwen | Q8_0 | `test_synth` | s1 | deployed | against the artefacts of record | +1.2 | 3 / 0 | 0.25 | [+0.0, +3.8] |
| Qwen | Q8_0 | `test_synth` | ctrl | deployed | rerun noise | +0.0 | 0 / 0 | 1 | [+0.0, +0.0] |
| Qwen | Q8_0 | `test_ood` | s1 | ctrl | the data change | +0.7 | 2 / 1 | 1 | [-1.3, +3.3] (items) |
| Qwen | Q8_0 | `test_ood` | s1 | deployed | against the artefacts of record | +0.7 | 2 / 1 | 1 | [-1.3, +3.3] (items) |
| Qwen | Q8_0 | `test_ood` | ctrl | deployed | rerun noise | +0.0 | 0 / 0 | 1 | [+0.0, +0.0] (items) |
| Llama | Q4_K_M | `test_golden` | s1 | ctrl | the data change | +0.0 | 3 / 3 | 1 | [-2.2, +2.4] |
| Llama | Q4_K_M | `test_golden` | s1 | deployed | against the artefacts of record | +0.5 | 4 / 3 | 1 | [-2.6, +3.2] |
| Llama | Q4_K_M | `test_golden` | ctrl | deployed | rerun noise | +0.5 | 2 / 1 | 1 | [-1.1, +2.2] |
| Llama | Q4_K_M | `test_synth` | s1 | ctrl | the data change | +1.2 | 4 / 1 | 0.375 | [-0.8, +4.2] |
| Llama | Q4_K_M | `test_synth` | s1 | deployed | against the artefacts of record | +0.8 | 5 / 3 | 0.727 | [-1.2, +3.8] |
| Llama | Q4_K_M | `test_synth` | ctrl | deployed | rerun noise | -0.4 | 2 / 3 | 1 | [-1.2, +0.0] |
| Llama | Q4_K_M | `test_ood` | s1 | ctrl | the data change | -0.7 | 2 / 3 | 1 | [-3.3, +2.0] (items) |
| Llama | Q4_K_M | `test_ood` | s1 | deployed | against the artefacts of record | -1.3 | 1 / 3 | 0.625 | [-4.0, +1.3] (items) |
| Llama | Q4_K_M | `test_ood` | ctrl | deployed | rerun noise | -0.7 | 3 / 4 | 1 | [-4.0, +2.7] (items) |
| Llama | Q8_0 | `test_golden` | s1 | ctrl | the data change | +1.0 | 4 / 2 | 0.688 | [-2.0, +4.4] |
| Llama | Q8_0 | `test_golden` | s1 | deployed | against the artefacts of record | +1.5 | 4 / 1 | 0.375 | [-1.1, +4.7] |
| Llama | Q8_0 | `test_golden` | ctrl | deployed | rerun noise | +0.5 | 1 / 0 | 1 | [+0.0, +1.6] |
| Llama | Q8_0 | `test_synth` | s1 | ctrl | the data change | +1.2 | 4 / 1 | 0.375 | [-0.8, +3.3] |
| Llama | Q8_0 | `test_synth` | s1 | deployed | against the artefacts of record | +0.8 | 4 / 2 | 0.688 | [-0.8, +2.9] |
| Llama | Q8_0 | `test_synth` | ctrl | deployed | rerun noise | -0.4 | 2 / 3 | 1 | [-1.7, +0.8] |
| Llama | Q8_0 | `test_ood` | s1 | ctrl | the data change | +1.3 | 4 / 2 | 0.688 | [-2.0, +4.7] (items) |
| Llama | Q8_0 | `test_ood` | s1 | deployed | against the artefacts of record | -4.0 | 0 / 6 | 0.0312 | [-7.3, -1.3] (items) |
| Llama | Q8_0 | `test_ood` | ctrl | deployed | rerun noise | -5.3 | 0 / 8 | 0.00781 | [-9.3, -2.0] (items) |

#### Qwen against Llama, every variant, at Q4_K_M

| Variant | Split | EM Qwen | EM Llama | only Qwen / only Llama | exact p | 95% CI Qwen - Llama (pp) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| deployed | `test_golden` | 0.935 | 0.910 | 5 / 0 | 0.0625 | [+0.5, +4.7] |
| deployed | `test_synth` | 0.954 | 0.958 | 3 / 4 | 1 | [-3.3, +1.7] |
| deployed | `test_ood` | 0.747 | 0.813 | 7 / 17 | 0.0639 | [-13.3, -0.7] (items) |
| ctrl | `test_golden` | 0.935 | 0.915 | 5 / 1 | 0.219 | [+0.0, +3.9] |
| ctrl | `test_synth` | 0.954 | 0.954 | 4 / 4 | 1 | [-2.1, +1.7] |
| ctrl | `test_ood` | 0.747 | 0.807 | 7 / 16 | 0.0931 | [-12.0, +0.0] (items) |
| s1 | `test_golden` | 0.925 | 0.915 | 7 / 5 | 0.774 | [-2.6, +4.6] |
| s1 | `test_synth` | 0.971 | 0.967 | 4 / 3 | 1 | [-0.8, +1.7] |
| s1 | `test_ood` | 0.767 | 0.800 | 11 / 16 | 0.442 | [-10.0, +3.3] (items) |

#### `test_golden` rotate items by direction cue, errors at Q4_K_M

| Cue | items | Qwen deployed | Qwen ctrl | Qwen s1 | Llama deployed | Llama ctrl | Llama s1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minus | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| counter-clockwise | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| clockwise | 5 | 5 | 5 | 4 | 5 | 5 | 4 |
| left | 4 | 0 | 0 | 0 | 0 | 0 | 1 |
| right | 7 | 6 | 6 | 6 | 7 | 7 | 5 |
| none | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| all | 22 | 11 | 11 | 10 | 12 | 12 | 10 |

#### `test_golden` rotate items by direction cue, errors at Q8_0

| Cue | items | Qwen deployed | Qwen ctrl | Qwen s1 | Llama deployed | Llama ctrl | Llama s1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minus | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| counter-clockwise | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| clockwise | 5 | 5 | 5 | 1 | 5 | 5 | 4 |
| left | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| right | 7 | 6 | 6 | 2 | 7 | 7 | 5 |
| none | 3 | 0 | 0 | 1 | 0 | 0 | 0 |
| all | 22 | 11 | 11 | 4 | 12 | 12 | 9 |

#### `test_synth` rotate items by direction cue, errors at Q4_K_M

| Cue | items | Qwen deployed | Qwen ctrl | Qwen s1 | Llama deployed | Llama ctrl | Llama s1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minus | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| counter-clockwise | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| clockwise | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| left | 9 | 1 | 1 | 0 | 1 | 0 | 0 |
| right | 6 | 6 | 6 | 5 | 6 | 6 | 4 |
| none | 1 | 0 | 0 | 0 | 1 | 1 | 0 |
| all | 20 | 8 | 8 | 6 | 9 | 9 | 6 |

#### `test_synth` rotate items by direction cue, errors at Q8_0

| Cue | items | Qwen deployed | Qwen ctrl | Qwen s1 | Llama deployed | Llama ctrl | Llama s1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minus | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| counter-clockwise | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| clockwise | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| left | 9 | 1 | 1 | 0 | 1 | 0 | 0 |
| right | 6 | 6 | 6 | 4 | 6 | 6 | 4 |
| none | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| all | 20 | 8 | 8 | 5 | 8 | 8 | 6 |

#### Errors and yaw sign errors on the two in-domain sets

Sign error: the prediction equals the gold command except that the yaw has the
opposite sign. Pooled over `test_golden` and `test_synth` (440 items).

| Model | Quant | Variant | errors | yaw sign errors |
| :--- | :--- | :--- | ---: | ---: |
| Qwen | Q4_K_M | deployed | 24 | 18 |
| Qwen | Q4_K_M | ctrl | 24 | 18 |
| Qwen | Q4_K_M | s1 | 22 | 15 |
| Qwen | Q8_0 | deployed | 24 | 18 |
| Qwen | Q8_0 | ctrl | 24 | 18 |
| Qwen | Q8_0 | s1 | 14 | 8 |
| Llama | Q4_K_M | deployed | 28 | 18 |
| Llama | Q4_K_M | ctrl | 28 | 18 |
| Llama | Q4_K_M | s1 | 25 | 14 |
| Llama | Q8_0 | deployed | 26 | 18 |
| Llama | Q8_0 | ctrl | 26 | 18 |
| Llama | Q8_0 | s1 | 21 | 13 |
