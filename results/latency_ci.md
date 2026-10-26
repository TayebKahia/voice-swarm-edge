### Language-model time per command, Q4_K_M (master article, Exp-1)

Scored trials only (warm-up excluded). Prompt cache on, as in the run of record.

#### cooled

| Measure | n | p50 [95% CI] (ms) | p95 [95% CI] (ms) |
| :--- | ---: | ---: | ---: |
| llama-3.2-1b-instruct-Q4_K_M prefill + decode | 60 | 1,351 [1,074, 1,548] | 2,282 [1,895, --] |
| llama-3.2-1b-instruct-Q4_K_M prefill | 60 | 268 [223, 286] | 457 [418, --] |
| llama-3.2-1b-instruct-Q4_K_M decode | 60 | 1,105 [810, 1,252] | 1,842 [1,656, --] |
| qwen2.5-0.5b-instruct-Q4_K_M prefill + decode | 60 | 647 [606, 762] | 1,072 [1,033, --] |
| qwen2.5-0.5b-instruct-Q4_K_M prefill | 60 | 38 [38, 161] | 329 [259, --] |
| qwen2.5-0.5b-instruct-Q4_K_M decode | 60 | 608 [455, 686] | 1,033 [880, --] |
| smollm2-360m-instruct-Q4_K_M prefill + decode | 60 | 475 [462, 600] | 894 [719, --] |
| smollm2-360m-instruct-Q4_K_M prefill | 60 | 27 [26, 112] | 241 [203, --] |
| smollm2-360m-instruct-Q4_K_M decode | 60 | 439 [336, 522] | 785 [676, --] |

#### uncooled

| Measure | n | p50 [95% CI] (ms) | p95 [95% CI] (ms) |
| :--- | ---: | ---: | ---: |
| llama-3.2-1b-instruct-Q4_K_M prefill + decode | 60 | 1,966 [1,553, 2,189] | 3,239 [2,660, --] |
| llama-3.2-1b-instruct-Q4_K_M prefill | 60 | 431 [399, 463] | 787 [749, --] |
| llama-3.2-1b-instruct-Q4_K_M decode | 60 | 1,505 [1,124, 1,694] | 2,459 [2,177, --] |
| qwen2.5-0.5b-instruct-Q4_K_M prefill + decode | 60 | 1,050 [973, 1,181] | 1,772 [1,569, --] |
| qwen2.5-0.5b-instruct-Q4_K_M prefill | 60 | 75 [51, 285] | 614 [465, --] |
| qwen2.5-0.5b-instruct-Q4_K_M decode | 60 | 923 [683, 1,064] | 1,519 [1,319, --] |
| smollm2-360m-instruct-Q4_K_M prefill + decode | 60 | 815 [727, 993] | 1,610 [1,269, --] |
| smollm2-360m-instruct-Q4_K_M prefill | 60 | 52 [36, 55] | 517 [391, --] |
| smollm2-360m-instruct-Q4_K_M decode | 60 | 701 [563, 858] | 1,339 [1,121, --] |

### Parse path on the 200 golden recordings (ingenieur article, Exp-2)

One sample per endpointed segment.

| Measure | n | p50 [95% CI] (ms) | p95 [95% CI] (ms) |
| :--- | ---: | ---: | ---: |
| Endpointing wait | 226 | 480 [480, 480] | 480 [480, 480] |
| Speech recognition | 226 | 1,246 [1,223, 1,262] | 1,449 [1,418, 1,522] |
| Prefill (no cache) | 226 | 571 [559, 599] | 751 [712, 791] |
| Decode | 226 | 561 [526, 566] | 915 [841, 950] |
| Validation and dispatch | 226 | 0 [0, 0] | 1 [1, 1] |
| End to end, after endpointing | 226 | 2,441 [2,311, 2,524] | 3,122 [3,034, 3,305] |
| End to end, including the endpointing wait | 226 | 2,921 [2,791, 3,004] | 3,602 [3,514, 3,785] |

### Reflex path with the take as the unit (ingenieur article, Exp-2)

First pass of each take. Replay difference: largest absolute difference in
latency from offset between the two passes of a take.

#### idle

| Measure | n | p50 [95% CI] (ms) | p95 [95% CI] (ms) |
| :--- | ---: | ---: | ---: |
| from keyword offset | 39 | 195 [155, 235] | 545 [436, --] |
|   spotter delay + frame quantisation | 39 | 190 [150, 230] | 540 [430, --] |
|   Pi compute + publish | 39 | 5 [5, 5] | 5 [5, --] |
| from keyword onset | 39 | 965 [885, 1,085] | 1,265 [1,225, --] |

Replay difference: max 0.3 ms, mean 0.1 ms over 39 takes.

- swarm_hold: detected 19/20 [0.751, 0.999]
- swarm_abort: detected 20/20 [0.832, 1.000]
- other class fired: 0/40 [0.000, 0.088]

#### loaded

| Measure | n | p50 [95% CI] (ms) | p95 [95% CI] (ms) |
| :--- | ---: | ---: | ---: |
| from keyword offset | 39 | 209 [164, 242] | 556 [436, --] |
|   spotter delay + frame quantisation | 39 | 190 [150, 230] | 540 [430, --] |
|   Pi compute + publish | 39 | 13 [7, 14] | 18 [16, --] |
| from keyword onset | 39 | 967 [886, 1,086] | 1,266 [1,226, --] |
| preemption recovery (both passes) | 78 | 601 [411, 802] | 1,195 [1,042, 1,505] |

Replay difference: max 12.4 ms, mean 5.8 ms over 39 takes.

- swarm_hold: detected 19/20 [0.751, 0.999]
- swarm_abort: detected 20/20 [0.832, 1.000]
- other class fired: 0/40 [0.000, 0.088]
- decode in flight at the trigger and cancelled: 39/39 [0.910, 1.000]
