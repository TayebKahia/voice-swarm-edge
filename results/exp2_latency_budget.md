### Latency budget versus measured

Raspberry Pi 5, cooled, governor `performance`, cores pinned (frame loop on core 0;
speech recognition and language model on cores 1-3). Real-time replay of recorded
audio: the parse path on the 200 golden utterances (clean), one sample per endpointed
segment; the reflex path on the author's 40 recorded keyword takes, idle and while the
parse path decodes, one detection per take (first pass) timed from the forced-aligned
keyword (P15); recovery over both passes. Nearest-rank percentiles; the verdict is on the p95.
Rows in bold are judged against a criterion of [Table](#tab:nonfunctional-requirements).
The bus is loopback on the Pi; the Wi-Fi hop is outside the budget.

| Stage | n | p50 | p95 | p99 | Target p95 | Verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Endpointing wait | 226 | 480 | 480 | 480 | 500 | MEETS |
| Speech recognition (whisper `tiny.en`, 3 threads) | 226 | 1,246 | 1,449 | 1,571 | 1,200 | MISSES |
| Language-model prefill | 226 | 570 | 751 | 849 | 250 | MISSES |
| Language-model decode | 226 | 561 | 915 | 984 | 1,100 | MEETS |
| Validation, state-machine check and dispatch | 226 | 0 | 1 | 3 | 50 | MEETS |
| **End to end, from end of speech** | 226 | 2,441 | 3,122 | 3,380 | 2,500 | MISSES |
| End to end, including the endpointing wait | 226 | 2,921 | 3,602 | 3,860 | 2,500 | MISSES |
| End to end, from start of speech | 226 | 6,343 | 10,086 | 12,322 | -- | -- |
| **Reflex path from keyword offset, idle** | 39 | 135 | 515 | 555 | 150 | MISSES |
| of which spotter delay and frame quantisation | 39 | 130 | 510 | 550 | -- | -- |
| of which Pi compute and publish | 39 | 5 | 5 | 6 | -- | -- |
| **Reflex path from keyword offset, loaded** | 39 | 144 | 518 | 564 | 150 | MISSES |
| of which spotter delay and frame quantisation | 39 | 130 | 510 | 550 | -- | -- |
| of which Pi compute and publish | 39 | 13 | 18 | 19 | -- | -- |
| Reflex path from keyword onset, idle | 39 | 1,055 | 1,355 | 1,375 | 850 | MISSES |
| Reflex path from keyword onset, loaded | 39 | 1,057 | 1,364 | 1,376 | 850 | MISSES |
| **Preemption recovery** | 78 | 601 | 1,195 | 1,505 | 300 | MISSES |

Prefill is the deployed parser's, with the prompt cache off; the budget's prefill
allowance assumes a cached prefix. Start-of-speech has no verdict: its 5,500 ms target is for a
3 s utterance. Split rate, cross-trigger matrix, pre-emption outcomes and the core-0 frame
budget: `results/exp2_analysis.md`.
