# Step 2: peer review of the Master and Ingénieur journal articles

Date: 2026-10-26. This is a report only; no article file was changed.

**What was reviewed.**
- Manuscript **A** (Master): `articles/master/main.tex`. Quantised small language models for grammar-constrained command parsing on a Raspberry Pi 5.
- Manuscript **B** (Ingénieur): `articles/ingenieur/main.tex`. A dual-path voice interface for drone-swarm formation control.
- Both were compiled with `changes` in `[final]` mode, so the clean text was reviewed: A has 12 pages including highlights, B has 13.
- Line numbers (`A:L123`, `B:L123`) refer to the `.tex` files at commit `6df5b95`.

**How the review was done.**
- **Phase A.** Six independent reviewers ran in parallel (Sonnet, high effort): a domain expert for A, a domain expert for B, a methodology and statistics reviewer, an adversarial reviewer, a presentation and editorial reviewer, and a venue scout.
- **Phase B.** I re-checked every claim that changes a recommendation against the manuscript, the raw data in `results/` and `data/`, and git history. I also re-resolved every suggested reference.
  - Claims that did not survive are dropped.
  - Claims a subagent could not confirm are marked "unverified" or left out.
- **Phase C.** This file.

**Tags.**
- **(E)** means existing material: it is already in the thesis, `results/`, `data/` or the repository and only needs bringing into the article. The location is given.
- **(N)** means new work. A rough effort is given in working time.
- Thesis sources are in `/home/kahia-tayeb/PFE/thesis/`.

**Facts established during consolidation.** They are not in either manuscript and they drive several recommendations. Each was recomputed or checked by me, not only by a subagent.

| # | Fact | Evidence |
|---|---|---|
| F1 | The golden set's errors sit almost entirely in one of its 12 template families. F106 (rotate, 22 items) holds 11 of Qwen2.5-0.5B's 13 Q4_K_M errors and 12 of Llama-3.2-1B's 18. Qwen scores 11/22 on F106 and 176/178 elsewhere. | `data/test_golden.jsonl` (`template_family`), `results/surface_b_preds/*Q4_K_M_test_golden.jsonl` |
| F2 | Every F106 error is a yaw-sign error on "clockwise" or "right". Of 120 rotate pairs in `data/train.jsonl`, only 6 contain "clockwise" (all with a negative yaw, so the convention is consistent but rare). The training set teaches counter-clockwise far more often than clockwise. | `data/train.jsonl` |
| F3 | A 12-family cluster bootstrap gives a Qwen-minus-Llama difference of about [+0.5, +4.7] pp (lower end near zero) and Qwen-minus-Danube about [+2, +12] pp. The item-level intervals in the article are much narrower. | Recomputed (10,000 replicates); the methodology reviewer's independent run agrees within rounding. |
| F4 | On the other two splits the Qwen-versus-Llama contrast reverses or vanishes. Synthetic: 3 items only Qwen gets right, 4 only Llama. Out-of-domain: 7 against 17, exact p = 0.064. | `results/surface_b_preds/` |
| F5 | The project protocol (commit `13847fc`, 2026-07-01) fixed one McNemar family of three model comparisons at α = 0.0167, with `test_synth` named as "the McNemar paired sample". The Q8_0-versus-Q4_K_M family and the switch of primary split to `test_golden` appear in commit `bfe1b5e` (2026-08-06), after the accuracy results (`54c3d4b`, 2026-08-01). | `git show 13847fc`, `git show bfe1b5e` (`eval/mcnemar.py`, `PRIMARY_SPLIT`) |
| F6 | The uncooled timing run came first (`5f60eaa`, 2026-08-05, "100% throttled"). The cooled run became the run of record ten days later (`98724ba`). | git log |
| F7 | In B's formation experiment, 423 of the 432 clamp interventions are the clamp separating drones that spawn closer than 0.8 m at t = 0. Line: 141 initial close pairs against 141 interventions. Wedge: 141 against 141. Circle: 141 against 150. The 48 seeds with an initial close pair are exactly the 144 trials with an intervention. B:L737–740 reads these as failures of the potential field. That reading is wrong. | `results/exp4.csv`, spawn regenerated from `swarm/env.py:97` |
| F8 | B's "78 triggers" are 39 distinct recorded takes, each replayed twice with near-identical latency. Idle replays differ by 0.1 ms on average. | `results/exp2_raw/` (methodology reviewer; consistent with B:L433) |
| F9 | The state-machine-to-controller link and a live demonstration chain now exist in the repository (`swarm/link.py`, `demo/`, commit `7bc56ab`, 2026-09-17; mic capture in `3546914`, 2026-10-04). B still says "not built" (B:L247, L343–345, L759) and reports no run that uses them. | git log, `swarm/`, `demo/` |
| F10 | 55 of the 200 golden labels occur verbatim in training. They are slot-free commands (takeoff, land, hover, abort, unknown), and every model gets them right. On the 145 unseen labels, Qwen's EM is 0.910. | `data/train.jsonl`, `data/test_golden.jsonl` |
| F11 | The public repository has no licence (GitHub API `license: null`), and its name ends in a hyphen (`voice-swarm-edge-`). | GitHub API |

---

# Report on Manuscript A

**Title:** Quantised small language models for grammar-constrained command parsing on a Raspberry Pi 5: an accuracy–efficiency comparison

## A.1 Summary

The authors fine-tune three instruction-tuned small language models (SmolLM2-360M, Qwen2.5-0.5B, Llama-3.2-1B) with LoRA, plus a size-matched control (H2O-Danube3-500M). The models map transcripts of drone-swarm commands to a ten-intent JSON schema. The three candidates are quantised to Q8_0 and Q4_K_M in GGUF, decoded by llama.cpp under a bounded GBNF grammar, and timed on a Raspberry Pi 5.

On a 200-item test set read by the first author and scored on reference transcripts:
- Qwen2.5-0.5B at Q4_K_M reaches an exact match of 0.935, against 0.910 for Llama-3.2-1B. The difference is not significant.
- The fp16-to-deployed pipeline changes exact match by at most 1.5 points, and removing the grammar produces one malformed output in 3,540 decodes.
- Throughput depends on cooling: 27.9 tokens/s cooled against 18.1 uncooled.
- No configuration meets the end-to-end latency, false-command or safe-failure targets.

The paper is carefully audited and unusually candid about its limits.

## A.2 Overall assessment and recommendation

**Major revision.**

The experiments are honest and internally consistent: every number checked against the raw data matched. The main reason is that the evidence base does not yet support the comparative claims, and the contribution is not differentiated from 2025–2026 work.
- The ranking of the two leading models rests on five items. Those items fall in one template family whose errors come from a single training-data gap (F1, F2), and the contrast reverses on the other test splits (F4).
- There is no informative baseline. The zero-shot condition scores 0.000 because its prompt never states the schema.
- The claim of novelty ("no study we reviewed…") is now contradicted by several close papers, including a fine-tuned small-model drone command parser published in 2026.

Most of the fixes are cheap reanalysis or short runs on existing infrastructure. With them, A is a credible Q1 submission to an applied-AI or embedded-systems journal.

## A.3 Strengths

1. **A real deployment-versus-reference design.** It uses paired McNemar tests, a template and token-ID parity check (A:L299–304), and verified workstation/Pi output identity (A:L306–308).
2. **Pre-set targets, with misses reported.** The selected configuration's failures are stated in the abstract.
3. **A useful error taxonomy.** 98.9% of in-domain errors would still dispatch an executable command (A:L587–592). That is an important and under-reported safety fact for small parsers.
4. **A concrete, practical thermal result** (Fig. 3). On the same board and artefact, the cooler decides whether the throughput floor is met.
5. **Open code, data, per-item predictions and per-trial timing.** Every table can be regenerated (`results/`, `eval/`).

## A.4 Major concerns

**A-M1. The statistical unit is the template family, not the item, and one family drives the ranking.**
- *Location:* A:L249–253, L341–353, L403–412, L516–519, L631–644; Table 4; abstract L77–79.
- *Why it matters:*
  - The golden set has 200 items but only 12 independent patterns, which the paper concedes in A:L351–353.
  - 11 of Qwen's 13 errors are in F106 (F1), and all of them are yaw-sign errors (F2).
  - Outside F106 both leading models are near ceiling (0.989 and 0.966).
  - The item-level McNemar p-values down to 10⁻⁷ and the bootstrap intervals overstate precision. The cluster interval for Qwen-minus-Llama barely excludes zero (F3).
- *Remedy:*
  - Add a per-family EM table.
  - Report family-clustered intervals (a cluster bootstrap, or a mixed-effects logistic model with a family random intercept) beside or instead of the item-level ones.
  - Cite clustered-error methodology (Miller, 2024; Eliasziw and Donner, 1991).
  - Report EM with and without F106.
  - Reword every "significant" in the abstract and Conclusion as item-level.
- *Tag:* **E**. Predictions are in `results/surface_b_preds/` and `train/kaggle_out/preds_*.jsonl`. About 3 hours.

**A-M2. No informative baseline.**
- *Location:* A:L410–412 (zero-shot EM 0.000 on all 590 items); A:L128–134, L200–203.
- *Why it matters:*
  - The zero-shot prompt is the one-line system prompt (A:L277–278), which never defines the schema, and the predictions were not kept. The result therefore shows that the models do not know the schema, not that fine-tuning is needed.
  - The corpus comes from 120 deterministic templates, so a reviewer will ask whether a rule-based parser or a small encoder does as well.
  - Without such a comparison, the paper cannot show that a language model is the right tool.
- *Remedy:* add, on the existing splits:
  - (i) few-shot or schema-in-prompt decoding under the same GBNF, no fine-tuning, for the same models;
  - (ii) a regex or template parser with number-word normalisation (`eval/norm.py` already has the normaliser);
  - (iii) a fine-tuned encoder for joint intent and slots (JointBERT-style; Chen et al., 2019), timed on the Pi;
  - (iv) optionally, a larger local model or a hosted model as an upper bound.
  
  Report where each fails. The result is informative either way.
- *Tag:* **N**. (i) about 1 day on the workstation, since the harness and grammar exist. (ii) 1–2 days. (iii) 2–3 days. (iv) about 0.5 day.

**A-M3. The novelty claim is out of date and the gap paragraph overreaches.**
- *Location:* A:L128–134, L185–192, L200–203, L657–659; contributions L140–153.
- *Why it matters:* "Constrained-decoding studies evaluate models not fine-tuned to the format" and "no study … scores fine-tuned language models below 1.5 B … under a grammar while timing … on a single-board computer" are now contradicted or narrowed by:
  - Ahmed et al. (2026): five small models fine-tuned for drone command parsing, 5,450 labelled commands, a rejection class, no cloud.
  - Nguyen et al. (2026): an edge drone voice-control system with a Raspberry Pi in the chain.
  - Wang et al. (2025, SLOT), which fine-tunes models and adds constrained decoding.
  - Chavan (2026) and Ray (2026): constrained decoding in small models.
  - Kavathekar et al. (2025): small-model function calling on an edge device.
  - TinyAgent and Octopus v2: fine-tuned, quantised on-device function callers.
  
  Full references are in A.7.
- *Remedy:*
  - Cite and differentiate these works.
  - State a search protocol for "no study we reviewed" (sources, dates, keywords).
  - Reposition the contribution as what is actually new: a paired-statistics audit of the complete fine-tune → quantise → GBNF → single-board-computer pipeline for one safety-relevant schema. That audit includes the thermal state and the abstention failure, and it isolates where the cost lies (not in quantisation or the grammar, but in latency, cooling and out-of-domain behaviour).
- *Tag:* **E** (text) plus reading, about 1 day.

**A-M4. The dominant error is a training-data coverage gap, and the paper's explanation for it is untested.**
- *Location:* A:L590–592, L684–689.
- *Why it matters:*
  - 40.4% of in-domain errors are yaw-sign errors. The paper offers an "untested explanation" that the transcript never contains a minus sign.
  - The data give a simpler, testable cause: only 6 of 120 training rotate pairs say "clockwise" (F2), and the system prompt never states the sign convention.
  - Because this one family decides the model ranking (A-M1), the gap matters for the headline result as well as the error analysis.
- *Remedy:*
  - (a) Report the clockwise/counter-clockwise counts in training (**E**, hours).
  - (b) Retrain at least Qwen2.5-0.5B with balanced rotation data, or with a direction enumeration (`"dir":"cw"/"ccw"`). Show the effect on EM and on the model ranking (**N**, 1–2 days).
  - Either way, present the finding as a lesson in schema and data design.

**A-M5. The provenance of the "confirmatory" tests is misdescribed, and the headline comparison holds on one split only.**
- *Location:* A:L343–346 ("Two confirmatory families of three comparisons were fixed before the results"); conclusion A:L711–713 ("at least as accurate as a model 2.5 times its size"); A:L631.
- *Why it matters:*
  - The protocol of 1 July 2026 fixed one family and named the synthetic set as the paired sample. The Q8_0/Q4_K_M family and the switch to the golden set as the primary split were introduced after the accuracy results (F5).
  - On the synthetic set Qwen and Llama split 3 to 4, and on the out-of-domain set Llama leads 17 to 7 (F4).
  - "At least as accurate" is a non-inferiority claim that was never tested.
  - If the public repository contains the protocol, a reviewer can find all of this.
- *Remedy:*
  - State that family 1 was fixed in the protocol (cite the commit or a tagged snapshot) and that family 2 and the choice of primary split are post hoc. Give the reason for the switch, which is real recorded speech rather than templates.
  - Report the Qwen-versus-Llama contrast on all three splits.
  - Replace the conclusion sentence, for example: "On the golden set it scored 2.5 points higher than a model 2.5 times its size (5 against 0 discordant items, not significant); on the out-of-domain set the larger model was ahead."
  - Note that Bonferroni correction per family does not control error across families; use Holm across all, or say so.
- *Tag:* **E**. Git history and `results/mcnemar.md`; about 2 hours.

**A-M6. The quantisation and grammar evidence is in the regime where a null result is expected.**
- *Location:* A:L285–297, L436–487, L646–667; Tables 2–3; abstract L80–83.
- *Why it matters:*
  - Only Q8_0 and Q4_K_M are studied, and Q4_K_M is the mildest aggressive level.
  - The "pipeline cost" changes precision, runtime, grammar and decoding at once.
  - "No established cost" is a non-inferiority statement made without a margin. Net differences hide churn: SmolLM2's fp16 and Q4_K_M outputs disagree on 21 items.
  - The grammar ablation pools 3,540 decodes from six correlated artefacts over the same 590 items. It is effectively one event in 590 items for one model.
  - A model fine-tuned to a rigid one-line schema already places nearly all its mass inside the grammar, so masking cannot change much (Park et al., 2024; Beurer-Kellner et al., 2024).
  - Grammar overhead on the Pi is not measured. Sampling cost under a grammar can depend on vocabulary size, which differs across the three tokenisers, so throughput differences may partly reflect grammar cost.
- *Remedy:*
  - (i) Add a ladder: HF fp16 → GGUF f16 under llama.cpp with and without the grammar → Q8_0 → Q4_K_M. The f16 GGUFs already exist as intermediates (`eval/check_parity_gguf.py`).
  - (ii) Add Q6_K, Q5_K_M, Q4_0, IQ4_XS, Q3_K_M and Q2_K to find where accuracy breaks.
  - (iii) Report paired differences with a stated equivalence margin (TOST, for example ±3 pp), plus flip rates.
  - (iv) Give the ablation per model with paired counts and a rule-of-three bound on the malformed rate (Hanley and Lippman-Hand, 1983).
  - (v) Repeat the grammar ablation where it can matter: Whisper transcripts (`results/exp0_transcripts.jsonl`), the few-shot models of A-M2, low-bit quantisations, and sampling at T > 0.
  - (vi) Time grammar on and off on the Pi.
  - Reword "cheap insurance" (A:L658–659) as "no difference detectable at this sample size, in distribution".
- *Tag:* (iv) **E** (`results/surface_b.csv`, `surface_b_nogrammar.csv`; 1 hour). (i)–(iii) and (v) **N**, about 1–2 days on the workstation. (vi) **N**, about 0.5 day on the Pi.

**A-M7. The timing and thermal protocol is thin, and the selection rule is circular.**
- *Location:* A:L310–323, L355–368, L521–564, L594–625; abstract L83–88; Fig. 2, Fig. 3, Fig. 5.
- *Why it matters:*
  - **Sample size.** There are 60 trials against 200 planned (A:L317). A p95 from 60 samples rests on three points, so the decode margin of 1,033 against 1,100 ms is not resolved.
  - **Q8_0 untimed.** Q8_0 was never timed, so half of the "accuracy–efficiency comparison" has no efficiency data.
  - **Cache regime.** Prefill is measured with a cached prefix that the deployed parser does not use (uncached p95 751 ms). The cached prefill is bimodal (Qwen p50 38.5 ms, p95 329 ms), so it measures cache hits.
  - **Thermal runs.** The thermal claim rests on one run per condition, on different days, with the uncooled run made first and the cooler added after it failed (F6). The clock was not logged.
  - **Circular selection rule.** The rule re-baselines the end-to-end budget to Qwen's own measured 3,122 ms (A:L617–619), so Qwen passes by construction. That figure also comes from the unpublished companion paper.
- *Remedy:*
  - Report order-statistic confidence intervals for every percentile.
  - Run 200 trials, time Q8_0, and time the deployed uncached configuration as the primary figure (**N**, about 1–2 days on the Pi).
  - Measure SmolLM2-360M end to end (**N**, about 3 hours), since it is the second non-dominated point.
  - Disclose the order of the thermal runs.
  - Replicate each thermal condition with `vcgencmd measure_clock` logging and record the cooler model and ambient temperature (**N**, about 1 day).
  - Define the selection rule on language-model-stage constraints that A measures itself: prefill + decode ≤ 1,350 ms, decode throughput, memory. Then A no longer depends on B and the rule is no longer circular (**E**; `results/exp1_cooled.csv`, `results/exp2_latency_budget.md`; 0.5 day of rewriting).
  - If the end-to-end rule stays, add the sentence "this budget is Qwen2.5-0.5B's own measurement, so it cannot test it".

**A-M8. Abstention is called the main shortfall but is not analysed, and the out-of-domain set is mostly author-written.**
- *Location:* A:L249–253, L566–592, L678–689; Fig. 4.
- *Why it matters:*
  - **Authorship of the set.** 138 of the 150 out-of-domain items were written by the authors: 75 authored, 38 authored in the MASSIVE taxonomy, and 25 authored stand-ins for recogniser output. Only 12 are real recogniser outputs on noise.
  - **Composition.** The four sources are equally weighted, so 0.19–0.39 is a stress-test rate, not an operating rate. For Qwen, the truncated fragments account for 20 of its 38 false commands.
  - **No mitigation tested.** The confidence gate is proposed (A:L688–689) but not implemented, and there is no risk–coverage analysis.
- *Remedy:*
  - (i) Describe the provenance of the out-of-domain set in §3.2 and report false-command rates per source (**E**; `data/ood.py`, `data/dataset_card.md`; hours).
  - (ii) Log token log-probabilities and plot risk–coverage or AURC for a min/mean token-probability gate, chosen on the validation set (**N**, 2–3 days; predictions must be re-run because the files keep only text).
  - (iii) Discuss the calibration and abstention literature (Stengel-Eskin and Van Durme, 2023; Wen et al., 2025; Larson et al., 2019).

**A-M9. The scope of the evidence is narrower than the claims.**
- *Location:* A:L249–258, L263–283, L691–703; abstract L76–77.
- *Why it matters:*
  - **Text only.** All accuracy is on reference transcripts. From audio, command recognition is 0.690 clean, and that gap is larger than any difference between models.
  - **Single reader.** The golden set is read by one non-native speaker who also wrote the templates. The protocol planned three to four speakers.
  - **Training.** Each model is trained once with one seed, and none had converged.
  - **Dated model set.** The models date from 2024 to early 2025. By October 2026 a reviewer will expect Qwen3-0.6B, Gemma 3 (270M/1B) and LFM2.
- *Remedy:*
  - (i) Score all six deployed artefacts on the existing Whisper transcripts of the golden set (`results/exp0_transcripts.jsonl`, clean and noisy) and report whether the ranking holds (**E** inputs, **N** hours of runs).
  - (ii) Add Qwen3-0.6B and Gemma-3-270M or 1B, and if possible a within-family size ladder (Qwen2.5-0.5B/1.5B, SmolLM2-135M/360M/1.7B) (**N**, 1–2 weeks including Pi timing).
  - (iii) Run 3 seeds per model and report mean ± SD (**N**, about 1 week elapsed on Kaggle).
  - (iv) If any audio claim remains in A, record 3–5 further speakers (**N**, 1–2 days; `data/record_session.py`).
  - At minimum, change "recorded commands" in the abstract (L76–77) to "a 200-item read test set, scored on reference transcripts".

**A-M10. The family control is too confounded to headline.**
- *Location:* abstract L78–80; highlight L94; contribution L144–145; A:L267–283, L424–434.
- *Why it matters:* the control (H2O-Danube3-500M) differs from Qwen2.5-0.5B in four ways besides family:
  - training session, with an unrecorded software image;
  - system-prompt rendering;
  - tokenisation of "drone";
  - convergence: it was still rising at epoch 3.

  It also has one seed and one model per family, and it was added post hoc. One pair cannot isolate "family".
- *Remedy:* demote it from the abstract, highlights and contribution list to an exploratory observation in Results. The cluster interval [+2, +12] pp (F3) supports keeping it as an observation (**E**, no work). To restore it as a contribution, retrain the control in the same session and software image with 2–3 seeds, and add a second matched pair (**N**, 2–3 days).

## A.5 Minor concerns

1. **Abstract (A:L70–89).** About 250 words, with roughly 25 numbers and undefined jargon: "re-baselining", "false-command target", "safe-failure target", "exact McNemar". Cut it to about 200 words: problem, method, three results, one limit, one implication. **E.**
2. **Highlight 3 (A:L94)** is the confounded family result. Replace it with the pipeline-cost result or the abstention failure. **E.**
3. **Intent F1 (A:L330)** is computed on the dispatched intent, after the validator's fallback (`eval/metrics.py`). Define it as such or report both. **E.**
4. **"`unknown` appears once in 2,640 in-domain decodes" (A:L587–588, L679).** The denominator includes items whose reference is `unknown`. The correct statement is 1 in 2,424 decodes of command items. **E.**
5. **Test choice (A:L341–343).** Use the exact McNemar test throughout; χ² tail p-values such as 1.1 × 10⁻⁷ have spurious precision. **E** (`results/mcnemar.csv` holds both).
6. **Bootstrap seed (A:L348).** Give the seed value (`eval/stats.py`) and the percentile rule. **E.**
7. **Accent range (A:L693).** "6.2% to 49.5%" quotes the extremes of buckets with n = 10 and 12. Quote buckets with n ≥ 20, or the pooled pattern. **E** (`results/table16_asr_speaker_sensitivity.md`).
8. **Anecdote (A:L122–124).** The motivating example rests on a 5-command run with a grammar that was not kept. Mark it as an illustration, or move it to §3.1 as a design note. **E.**
9. **"Golden test set".** This is idiosyncratic. Use "human-read test set" and explain that the reader wrote the templates. Define on first use: "reference/deployed surface", "artefact", "prefill", "Q4_K_M", "non-dominated", "dispatched". **E.**
10. **Noise partitions (A:L244).** State that augmentation noise and evaluation noise are disjoint partitions (DREGON first 70% / last 30% of a flight; ESC-50 folds 1–4 / 5; `data/mix_noise.py:166–179`). Also say they come from the same flight, which matters for B. **E.**
11. **Decoding settings.** Report them in one place: temperature 0, n_predict 96, context size, threads, cache flag per experiment. Also state that no importance matrix was used. **E.**
12. **Power and energy.** These are not measured, yet they are a first-order metric for an onboard parser. Add one table with a USB power meter (**N**, about 1 day), or list it as a limitation (**E**).
13. **Duplicate gap statement.** The gap is stated twice (A:L128–134 and L200–203). Keep one. Compress "Intent and slot corpora" (A:L194–198) to two sentences. **E.**
14. **Limitations (A:L691–703).** This is one 270-word paragraph. Turn it into a list ordered by severity. **E.**
15. **Figures and tables.**
    - Fig. 1 duplicates Tables 1–2 and has no confidence-interval bars. Redraw it as a dot-and-interval plot with clustered intervals.
    - Table 3 (two rows) repeats the text. Cut it, or make it per-model (A-M6).
    - Merge Table 2 into Table 1.
    - Fig. 2 duplicates Table 1 and Fig. 5.
    - Move Table 4 to the supplement.
    - In Fig. 3, the white "18.1" label on orange has low contrast.
    - In Fig. 5, drop the end-to-end line, since the axis does not measure it.
    
    **E** (`eval/plots.py`; about half a day).
16. **Token count (A:L224–226).** The 16.0 vs 19.6 token figure is for one tokeniser. Give it for all three. **E** (`results/wire_format_tokens.md`).
17. **Data availability (A:L732–737).**
    - Add a licence (F11) and archive a release with a DOI (Zenodo).
    - Fix or explain the trailing hyphen in the repository URL.
    - State that only the Qwen Q4_K_M artefact is released, and release the other GGUFs and the LoRA adapters if licences allow. Llama 3.2 weights are gated.
    - State that DREGON and ESC-50 (CC BY-NC) mixed audio is regenerated rather than distributed.
    - Add a consent line for the author's voice recordings.
    
    **E**, about 2 hours.
18. **Generative-AI declaration (A:L739–742).** "In order to assist with the writing" is vague. State the scope (for example drafting and language editing). Restore "reviewed and edited", as in Elsevier's model statement. **E.**
19. **Blocking placeholders.**
    - Placeholders remain for e-mails, the corresponding author and ORCID (A:L54–68, L40).
    - `changes` is still in `[draft]` mode (A:L22). Switch it to `[final]` and delete the audit macros before submission.

    **E.**
20. **Graphical abstract.** None. The target journal may request one. Assemble it from Fig. 5 and Fig. 3 (**E**, 2–3 hours).

## A.6 What to add, remove and restructure

**Add**
- Baselines: few-shot + grammar, rule-based parser, small encoder, upper-bound model (A-M2).
- A per-family table, clustered intervals and the F106 analysis (A-M1, A-M4).
- The Qwen-versus-Llama contrast on all three splits, and the true provenance of each statistical family (A-M5).
- A quantisation ladder including GGUF f16 and lower bit widths, with an equivalence margin (A-M6).
- Order-statistic CIs for latencies, Q8_0 timing, deployed uncached timing, and SmolLM2 end to end (A-M7).
- Out-of-domain provenance, a per-source breakdown and a risk–coverage curve (A-M8).
- Scoring on the existing Whisper transcripts, plus Qwen3-0.6B and Gemma-3 rows (A-M9).
- 2025–2026 related work (A.7) and a graphical abstract.

**Remove**
- The family control from the abstract, highlights and contribution list (A-M10).
- "Cheap insurance", "at least as accurate", and "thermal state decides" as unqualified claims (A-M5, A-M6, A-M7).
- Table 3 (two rows) and Fig. 2 (redundant), and the second gap statement.
- The dependence on the companion's 3,122 ms figure for the selection rule (A-M7).
- Hyperparameter detail from the running text (A:L270–277); move it to a table or the supplement.

**Restructure**
- Results order: (1) accuracy with its statistics, (2) cost of quantisation and grammar, (3) board performance (latency, memory, thermal), (4) abstention, (5) selection. Table 4 is currently used before it appears.
- Replace the selection-rule prose (A:L608–625) with a configuration × constraint table (met / missed / not measured).
- Contributions as four verifiable claims, for example:
  - "A 0.49 B model is not shown to be less accurate than a 1.24 B one on 590 items…"
  - "The fp16-to-Q4_K_M pipeline under grammar costs at most three items…"
  - "Cooling decides whether the throughput floor is met in our runs…"
  - "None of six configurations abstains adequately…"
- Title options:
  - "Fine-tuned small language models for on-board drone-swarm command parsing: quantisation, grammar-constrained decoding and thermal limits on a Raspberry Pi 5"
  - "Does model size matter for edge command parsing? Fine-tuned, quantised and grammar-constrained language models on a Raspberry Pi 5"
- Add "quantization" (US spelling), "drone swarm" and "Raspberry Pi" to the keywords.

## A.7 Missing references

Each was resolved during this review through arXiv, Crossref, ACL Anthology, PMLR or the publisher page. None is in `articles/master/references.bib`. Items marked "arXiv only" have no confirmed peer-reviewed version.

**Closest competitors (must cite)**
1. Ahmed, O., Stanziano, J., Sudharsanan, R., Kondragunta, R., Khalid, A., 2026. Voice-Controlled Unmanned Aerial System Using a Fine-Tuned Small Language Model for Real-Time Command Parsing. Artificial Super Intelligence (ASI) Conference 2026, Kennesaw State University. https://digitalcommons.kennesaw.edu/asi/2026/proceedings/2. **Cite at** A:L163–171 and A:L200–203; differentiate it in the gap statement.
2. Nguyen, D.-T., Tran, P.-L., Nguyen, C.-N., Tran, H.-P., Tran, D.-P., 2026. LLM and NLP-based drone control: a distributed edge computing architecture for Vietnamese. FME Transactions 54(3), 402–410. https://doi.org/10.5937/fme2603402N. **Cite at** A:L163–171. It uses an 8B model on a Jetson with recognition on a Raspberry Pi.
3. Simões, L.E.P., Rodrigues, L.B., Silva, R.M., da Silva, G.R., 2024. Evaluating Voice Command Pipelines for Drone Control: From STT and LLM to Direct Classification and Siamese Networks. arXiv:2407.08658. https://arxiv.org/abs/2407.08658. **Cite at** A:L163–171 as the non-LLM comparator (already cited by B).

**Constrained decoding and small-model structured output**

4. Chavan, A., 2026. Constrained Decoding Eliminates Structural Failures in Small LLMs but Reveals a Scale-Dependent Semantic Gap. arXiv:2609.23742 (arXiv only). https://arxiv.org/abs/2609.23742. **Cite at** A:L128–134, L185–192 and in the Discussion at L646–659.
5. Ray, J., 2026. The Constraint Tax: Measuring Validity-Correctness Tradeoffs in Structured Outputs for Small Language Models. arXiv:2605.26128 (arXiv only). https://arxiv.org/abs/2605.26128. **Cite at** A:L185–192 and L654–659. Its accuracy loss without fine-tuning contrasts with the null result here.
6. Wang, D.Y.-B., Shen, Z., Mishra, S.S., Xu, Z., Teng, Y., Ding, H., 2025. SLOT: Structuring the Output of Large Language Models. arXiv:2505.04016. https://arxiv.org/abs/2505.04016. **Cite at** A:L131–132, L191–192 and L657–658. It combines fine-tuning with constrained decoding, which contradicts "not fine-tuned to the target format".
7. Geng, S., Cooper, H., Moskal, M., Jenkins, S., Berman, J., Ranchin, N., West, R., Horvitz, E., Nori, H., 2025. JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models. arXiv:2501.10868. https://arxiv.org/abs/2501.10868. **Cite at** A:L185–192. It compares engines, including llama.cpp.
8. Dong, Y., Ruan, C.F., Cai, Y., Xu, Z., Zhao, Y., Lai, R., Chen, T., 2025. XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models. MLSys 2025. https://arxiv.org/abs/2411.15100 (MLSys listing: https://mlsys.org/virtual/2025/3235). **Cite at** A:L185–188.
9. Beurer-Kellner, L., Fischer, M., Vechev, M., 2024. Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation. ICML 2024, PMLR 235. https://proceedings.mlr.press/v235/beurer-kellner24a.html. **Cite at** A:L652–655, on when constraints distort the distribution.
10. Ugare, S., Suresh, T., Kang, H., Misailovic, S., Singh, G., 2024. SynCode: LLM Generation with Grammar Augmentation. arXiv:2403.01632. https://arxiv.org/abs/2403.01632. **Cite at** A:L185–188.
11. Raspanti, F., Ozcelebi, T., Holenderski, M., 2025. Grammar-Constrained Decoding Makes Large Language Models Better Logical Parsers. ACL 2025 Industry Track. https://aclanthology.org/2025.acl-industry.34/. **Cite at** A:L185–192 and with baseline (i) of A-M2.

**Function calling and tool use with small models**

12. Patil, S.G., Mao, H., Yan, F., Ji, C.C.-J., Suresh, V., Stoica, I., Gonzalez, J.E., 2025. The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. ICML 2025, PMLR 267. https://proceedings.mlr.press/v267/patil25a.html. **Cite at** A:L119–126 and with the abstention discussion (BFCL includes irrelevance detection).
13. Kavathekar, I., Donakanti, R., Kumaraguru, P., Vaidhyanathan, K., 2025. Small Models, Big Tasks: An Exploratory Empirical Study on Small Language Models for Function Calling. arXiv:2504.19277. https://arxiv.org/abs/2504.19277. **Cite at** A:L128–134 and L200–203.
14. Erdogan, L.E., Lee, N., Jha, S., Kim, S., Tabrizi, R., et al., 2024. TinyAgent: Function Calling at the Edge. arXiv:2409.00608. https://arxiv.org/abs/2409.00608. **Cite at** A:L163–171.
15. Chen, W., Li, Z., 2024. Octopus v2: On-device language model for super agent. arXiv:2404.01744. https://arxiv.org/abs/2404.01744. **Cite at** A:L163–171.
16. Lin, Q., Wen, M., Peng, Q., Nie, G., Liao, J., et al., 2024. Hammer: Robust Function-Calling for On-Device Language Models via Function Masking. arXiv:2410.04587. https://arxiv.org/abs/2410.04587. **Cite at** A:L678–689 (irrelevance detection).
17. Roshaan, M., Aaliyan, M., Zubair, S.B., 2026. Compression Breaks Discipline: Quantization Increases Missed Tool Calls in Small Language Models, but Not Mid-Size Ones. NeurIPS 2026 workshop. https://neurips.cc/virtual/2026/171153. **Cite at** A:L178–183, as evidence that quantisation can hurt small models on structured tasks.

**Quantisation and edge measurement**

18. Zheng, X., Li, Y., Chu, H., Feng, Y., Ma, X., et al., 2025. An Empirical Study of Qwen3 Quantization. arXiv:2505.02214. https://arxiv.org/abs/2505.02214. **Cite at** A:L178–183.
19. Li, S., Ning, X., Wang, L., Liu, T., Shi, X., et al., 2024. Evaluating Quantized Large Language Models. ICML 2024. https://arxiv.org/abs/2402.18158. **Cite at** A:L178–183.
20. Lu, Z., Li, X., Cai, D., Yi, R., Liu, F., et al., 2024. Small Language Models: Survey, Measurements, and Insights. arXiv:2409.15790. https://arxiv.org/abs/2409.15790. **Cite at** A:L110–117 and L173–177. It includes on-device latency and memory measurements.
21. Ardakani, M., Malekar, J., Zand, R., 2025. LLMPi: Optimizing LLMs for High-Throughput on Raspberry Pi. CVPR 2025 Workshops (EDGE). https://openaccess.thecvf.com/content/CVPR2025W/EDGE/html/Ardakani_LLMPi_Optimizing_LLMs_for_High-Throughput_on_Raspberry_Pi_CVPRW_2025_paper.html. **Cite at** A:L173–177.
22. Tummalapalli, P., Arayakandy, S., Pal, R., Kundan, K., 2026. LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load. arXiv:2603.23640. https://arxiv.org/abs/2603.23640. **Cite at** A:L669–677. Thermal effects under sustained load are known, so keep the thermal claim modest.

**Candidate models**

23. Yang, A., et al. (Qwen Team), 2025. Qwen3 Technical Report. arXiv:2505.09388. https://arxiv.org/abs/2505.09388. **Cite at** A:L263–268.
24. Gemma Team, 2025. Gemma 3 Technical Report. arXiv:2503.19786. https://arxiv.org/abs/2503.19786. **Cite at** A:L263–268.
25. Amini, A., Banaszak, A., Benoit, H., Böök, A., Dakhran, T., et al. (Liquid AI), 2025. LFM2 Technical Report. arXiv:2511.23404. https://arxiv.org/abs/2511.23404. **Cite at** A:L263–268.
26. Belcak, P., Heinrich, G., Diao, S., Fu, Y., Dong, X., et al., 2025. Small Language Models are the Future of Agentic AI. arXiv:2506.02153. https://arxiv.org/abs/2506.02153. **Cite at** A:L110–117, for motivation.

**Baselines, abstention and statistics**

27. Chen, Q., Zhuo, Z., Wang, W., 2019. BERT for Joint Intent Classification and Slot Filling. arXiv:1902.10909. https://arxiv.org/abs/1902.10909. **Cite at** A:L194–198 and with baseline (iii) of A-M2.
28. Larson, S., et al., 2019. An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1131/. **Cite at** A:L566–592.
29. Stengel-Eskin, E., Van Durme, B., 2023. Calibrated Interpretation: Confidence Estimation in Semantic Parsing. TACL. https://arxiv.org/abs/2211.07443. **Cite at** A:L688–689 (confidence gate).
30. Wen, B., Yao, J., Feng, S., Xu, C., Tsvetkov, Y., Howe, B., Wang, L.L., 2025. Know Your Limits: A Survey of Abstention in Large Language Models. TACL 13, 529–556. https://doi.org/10.1162/tacl_a_00754. **Cite at** A:L678–689.
31. Miller, E., 2024. Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. arXiv:2411.00640. https://arxiv.org/abs/2411.00640. **Cite at** A:L338–353 for clustered standard errors.
32. Eliasziw, M., Donner, A., 1991. Application of the McNemar test to non-independent matched pair data. Statistics in Medicine 10(12), 1981–1991. https://doi.org/10.1002/sim.4780101211. **Cite at** A:L341–353.
33. Dietterich, T.G., 1998. Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. Neural Computation 10(7), 1895–1923. https://doi.org/10.1162/089976698300017197. **Cite at** A:L697–699: McNemar does not capture training-run variance.
34. Dror, R., Baumer, G., Shlomov, S., Reichart, R., 2018. The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing. ACL 2018. https://aclanthology.org/P18-1128/. **Cite at** A:L338–353.
35. Hanley, J.A., Lippman-Hand, A., 1983. If Nothing Goes Wrong, Is Everything All Right? Interpreting Zero Numerators. JAMA 249(13), 1743–1745. https://doi.org/10.1001/jama.1983.03330370053031. **Cite at** A:L484–487 for the rule-of-three bound.

## A.8 Suggested venues

Venue facts were checked on Elsevier's own pages where possible (scope text, desk-rejection rules, and the APC list dated 05-Oct-2026). Impact factors and quartiles come from secondary sources (LetPub, bioxbio) reflecting JCR 2025, and could not be read on ScienceDirect, which blocks automated access. **Confirm them on each journal's Guide for Authors before submitting.**

| Rank | Journal | Quartile / IF (secondary) | APC (hybrid) | Why it fits | What A must change |
|---|---|---|---|---|---|
| 1 | **Engineering Applications of Artificial Intelligence** (Elsevier) | JCR Q1 (AI; Automation & Control), IF ≈ 9.0 | US$3,040; free subscription route | Scope is "practical application of AI methods"; robotics is a listed focal point. | Verified desk-reject rules (https://shop.elsevier.com/journals/engineering-applications-of-artificial-intelligence/0952-1976): **single-column format**, the abstract must separate the AI contribution from the engineering application, and no undefined acronyms in title or abstract (Q4_K_M, GBNF, EM, fp16 all appear now). Also: validation "using public data sets", so archive the dataset with a DOI. The baselines of A-M2 are near-essential here, because the journal expects a clear AI contribution. |
| 2 | **Computers and Electrical Engineering** (Elsevier) | JCR Q1 (Hardware & Architecture), IF ≈ 5.5 | US$3,270 | Broad applied-computing scope including AI on hardware; lower risk. | Frame the paper as an edge-deployment evaluation. User-reported review times are long (secondary source). |
| 3 | **Journal of Systems Architecture** (Elsevier) | JCR Q1 (Hardware & Architecture), IF ≈ 5.3 | US$2,260 | Embedded and real-time systems "from microarchitecture to application"; the deployment study fits well. | Use it only if B goes elsewhere (see X-3). Reviewers may ask for a second board or runtime. |
| Alt. | IEEE Access (non-Elsevier, gold open access) | IF 4.2 (stated by IEEE) | APC not confirmed on IEEE's page | Fast, wide scope. | Lower prestige; use as a fallback only. |

**Avoid**
- **Expert Systems with Applications.** It is Q1, but its scope page states: "The journal no longer considers papers that contain applications to military/defense systems". Verified at https://shop.elsevier.com/journals/expert-systems-with-applications/0957-4174. A drone-swarm paper invites that objection.
- **MDPI *Drones*.** Not recommended for a Q1 objective.

---

# Report on Manuscript B

**Title:** A dual-path voice interface for drone-swarm formation control on a Raspberry Pi 5: reflex preemption of a small language model

## B.1 Summary

The authors present a voice interface for five simulated drones whose speech and language stages run on a Raspberry Pi 5.
- **Parse path.** A voice-activity endpointer, whisper.cpp tiny.en, the fine-tuned Qwen2.5-0.5B under a grammar from the companion paper, and a semantic validator.
- **Reflex path.** A two-class keyword spotter that publishes "hover" or "abort" without transcription and cancels any decode in progress.
- **Membership rule.** Two tests: the command's delay must change the outcome, and a false trigger must make the swarm do less. The rule admits two of the ten intents to the reflex path.

Results:
- Every reflex trigger that met a decode cancelled it.
- Both paths miss their latency budgets: reflex 545 ms against 150 ms at p95; parse 3,122 ms against 2,500 ms.
- Recognition under rotor noise misses its targets.
- Formation control is evaluated separately, without voice, in a kinematic simulator.

The authors claim only the membership rule and the reflex-path budget as contributions.

## B.2 Overall assessment and recommendation

**Reject in its current form, with encouragement to resubmit.**

The writing is honest, the latency attribution is careful, and the question of which commands may bypass interpretation is a good one. But a Q1 venue will not accept the paper on its present evidence:
- **The membership rule is argued, not evaluated.** Its premise that abort "does less" is debatable.
- **The headline component misses its budget by 3.6×.** The reason is inside the design (spotter delay), not the platform.
- **There is no baseline**, not even the single-path system the paper argues against.
- **The system was never run from voice to vehicle**, although the code for that link now exists (F9).
- **The formation experiment adds no information and contains a misreading** (F7).
- **Reflex error rates were never measured in noise**, the condition the path is designed for.

Most of these can be fixed in two to four weeks with the existing infrastructure. If the fixes cannot be made, see X-2 for the merge option.

## B.3 Strengths

1. **A well-posed and under-studied question.** Which commands may bypass interpretation? The paper states the condition under which a third class could be admitted (B:L347–349).
2. **A correct, non-obvious ordering rule.** The sequence number is reserved at acceptance, not at completion (B:L291–302), and correctness (ordering) is separated cleanly from optimisation (cancellation).
3. **Per-stage latency attribution.** It separates spotter delay from the board's 6–18 ms (Table 2, Fig. 2) and is genuinely informative.
4. **Statistical care on small counts.** Exact and Poisson intervals and a "not demonstrated" verdict (B:L441–444), and the authors disclose the double fit of the keyword heads (B:L399–401).
5. **Candid limitations and full raw records** in the public repository.

## B.4 Major concerns

**B-M1. The membership rule, the paper's only claimed contribution, is an argument rather than a result, and its safety premise is contestable.**
- *Location:* B:L127–158, §3.3 (B:L304–349), Table 1, B:L365–371.
- *Why it matters:*
  - The rule is a two-column judgement over ten intents the authors defined themselves.
  - "Abort … a descent in place with the motors stopped on ground contact" is not obviously "less". A false abort lands five vehicles wherever they are, possibly onto people, obstacles or each other, and latches a state that only a manual reset clears.
  - The "time saved" test uses 2,350 ms × 2.0 m/s = 4.7 m, compared with inter-drone spacing (B:L334–335). That is not a hazard criterion: collisions are defined at 0.35 m, and the 2.0 m/s ceiling does not apply to formation changes (B:L762–763).
  - The false-accept (≤1/h) and false-reject (≤0.10) budgets are not derived from any hazard analysis.
- *Remedy:*
  - (i) Add a short hazard analysis (STPA, Leveson 2012, or FMEA) of the four outcomes: false hold, false abort, missed hold, missed abort. Revisit whether abort belongs on the reflex path.
  - (ii) Derive the "time saved" test from stopping distance (reaction + v·t + v²/2a) at the speeds actually allowed.
  - (iii) Turn the rule into a checkable property, for example: reflex outputs must be minimal in a partial order on commands by kinetic consequence given the current state.
  - (iv) Position the reflex path as a runtime-assurance safe channel: Black-Box Simplex (Mehmood et al., 2022), ASTM F3269-21, RoboGuard (Ravichandran et al., 2025). State that a voice stop supplements, and does not replace, the autopilot failsafe and a hardware emergency stop (ISO 13850:2015).
  - (v) Model-check the ordering rule and the five-state machine (TLA+ or UPPAAL), or run randomised interleaving tests (`runtime/bus.py`, `swarm/fsm.py`).
- *Tag:* **N**. (i)–(iii) 3–5 days; (iv) 1 day of reading and writing; (v) 3–5 days.

**B-M2. The reflex path misses its budget, and the cause is inside the design.**
- *Location:* Table 2 (B:L494–501), B:L530–537, L701–710; abstract L79–80.
- *Why it matters:*
  - p95 is 545 ms against 150 ms, and 1,265 ms from onset against 850 ms. The paper attributes 540 ms to "spotter delay plus frames".
  - The offset anchor is an energy estimate. 10 of 156 detections come before it, by up to 235 ms (B:L533–534), so the metric is not reliable at the 150 ms scale.
  - The threshold sits at 0.999, where scores saturate (the operating curve is flat above 0.88, B:L570–571). The 16-embedding (about 2 s) window and the 1 s debounce (B:L392–393) are plausible causes, but delay against threshold was never measured.
  - The p95 is set by two takes (w21, w33).
  - The paper's own comparison shows MIRA's gate is faster from onset: 466 ms against 965 ms median (B:L752).
- *Remedy:*
  - Anchor the offset by forced alignment, or synthesise takes with known end samples.
  - Measure the delay–threshold curve.
  - Shorten the context or debounce, or train with a latency-aware objective (Jose et al., 2022). Then re-measure against 150 ms.
  - If the budget still cannot be met, say so and justify 150 ms against human reaction time (Woods et al., 2015), or revise the budget openly.
- *Tag:* the delay–threshold curve is **N**, hours (`eval/wake_real.py`). Retraining and re-measuring is **N**, 1–2 weeks (`eval/exp2.py` exists).

**B-M3. No baseline for the dual-path claim.**
- *Location:* abstract L71–72, B:L116–123, L713–715, L745–753.
- *Why it matters:*
  - The premise ("too slow to stop a drone swarm") is supported only by comparing the parse path on long golden-set commands (3,602 ms) with the reflex path on short keyword takes (545 ms). The anchors and utterances differ.
  - No other system was run. Lim et al., CommandSwarm and SkySim are compared "by design".
  - A reviewer will also ask for the obvious alternatives:
    - a hold triggered by voice onset (MIRA-style), which needs no keyword and passes the paper's own membership rule;
    - a rule or regex parser on the Whisper transcript;
    - grammar-constrained speech recognition with no language model.
- *Remedy:* on identical audio and anchors, measure:
  - (a) the 40 stop takes through the parse path (single-path baseline);
  - (b) a voice-onset hold;
  - (c) a regex or fuzzy parser on the transcript, for command recognition rate and latency on the golden set;
  - (d) optionally, a grammar-constrained recogniser (Vosk-style) and a cloud-model path.
- *Tag:* (a) **N**, 2–3 hours on the Pi (harness and takes exist). (b) and (c) **N**, 2–3 days (`runtime/vad.py`, `runtime/stt.py`). (d) **N**, about 1 week.

**B-M4. The system was never run end to end, although the code for it now exists.**
- *Location:* Fig. 1 (B:L246–247, "not built"); B:L343–345, L462, L754–761; Table 4 rows "Offline operation" and "Live demonstration"; title.
- *Why it matters:*
  - No hold or abort ever reached a simulated vehicle.
  - No run joined speech to formation.
  - No live-microphone run, no run with networking disabled, and no Wi-Fi hop with packet loss were performed. The stop travels as an unacknowledged UDP datagram.
  - The repository already contains the state-machine-to-controller link, the live demonstration chain and microphone capture (F9). The article is stale relative to the code, and the most damaging limitation is cheap to remove.
- *Remedy:*
  - Run voice → state machine → controller on PyFlyt, live microphone, over Wi-Fi, with and without networking.
  - Report latency from utterance onset to vehicle arrest, stopping distance and time to rest for hold and abort.
  - Inject datagram loss (`tc netem`) and add acknowledgement or repetition for stop messages.
- *Tag:* **E** for the code (`swarm/link.py`, `demo/`, commits `7bc56ab` and `3546914`). **N** for the runs: 1–3 days, plus 1–2 days for loss injection.

**B-M5. The reflex evidence is pseudo-replicated and comes from one speaker in clean audio.**
- *Location:* abstract L78; highlight L92; B:L430–444, L539–545, L562–571; Table 4.
- *Why it matters:*
  - The "78 triggers" are 39 distinct takes replayed twice, deterministically (F8). With 0 confusions in 39 takes, the 95% upper bound on the confusion rate is about 9%; cancellation success is ≥ 91%.
  - False rejects come from 20 takes per class by the author. False accepts come from 1 event in 0.429 h of clean read speech (interval [0.06, 12.98]/h).
  - The keyword spotter was never tested under rotor noise, the condition it is designed for. The rotor-noise sweep covers only the parse path.
  - One live false hover during golden playback is unexplained (B:L543–545).
- *Remedy:*
  - Make the take the unit throughout: n = 39, x/20 per class with exact intervals. Present the second replay as a determinism check (**E**, about 2 hours; `results/exp2_raw/`).
  - Evaluate false rejects and false accepts with DREGON mixed at 20, 10 and 5 dB (**N**, about 1 day; `data/mix_noise.py`).
  - Extend the ambient stream to ≥ 10 hours from public corpora (**N**, 2–3 days).
  - Record keyword takes from ≥ 5 speakers (**N**, 1–2 weeks with consent).
  - Investigate the live false hover by replaying the recording (**N**, about 0.5 day), or move it to Limitations.

**B-M6. The formation-control section misreads its own data and carries no information.**
- *Location:* title; abstract L84–86; §3.6, B:L462–469, §4.5 (B:L645–656); Discussion B:L737–743; Table 4 rows "Collisions" and "Formation accuracy"; Figs. 5–6.
- *Why it matters:*
  - B:L737–740 states that "the field alone let pairs inside 0.80 m 432 times". In fact 423 of the 432 interventions separate drones that spawn too close at t = 0 (F7). In the six trials without close spawns, the field kept 0.82–0.89 m unaided.
  - Collisions are defined after the clamp, so zero collisions is true by construction (the code says so), yet Table 4 lists it as "met".
  - Formation accuracy is 1.000 in every trial.
  - The 150 trials are 50 seeds × 3 shapes, so the ANOVA (F = 3,500.7, p = 10⁻¹²⁴) tests a deterministic simulator.
  - Formations were never commanded by voice, yet "formation control" leads the title.
- *Remedy:*
  - Either cut the section to a short supplementary sanity check and drop "formation control" from the title, or replace it with the voice-to-vehicle run of B-M4.
  - In any case, correct the Discussion. Mark the collisions row "not informative (zero by construction)".
  - Run a clamp-off ablation and a wider spawn to show whether the field alone separates the vehicles.
  - Drop the ANOVA and Tukey in favour of medians with intervals.
- *Tag:* correction **E** (`results/exp4.csv`). Clamp-off and spawn variants **N**, about 1 hour each (`eval/exp4.py`).

**B-M7. The parse path, not the reflex path, carries the larger false-command risk, and the paper does not report it.**
- *Location:* §3.4 (B:L354–371), B:L726–735; absent from the Results.
- *Why it matters:*
  - The membership rule protects against false reflex triggers. But any speech the endpointer accepts is transcribed and parsed, and there is no wake word, push-to-talk or confirmation for takeoff, move, rotate or formation.
  - The deployed parser dispatches a non-`unknown` command on 25.3% of out-of-domain inputs (companion data), against a ≤ 0.05 target.
  - B reports only the safe-failure rate under noise (0.014–0.039).
- *Remedy:*
  - Report the out-of-domain false-command rate in B (**E**; `results/nfr18_false_command.md`, `results/limitation_abstention.md`).
  - Add a gate for movement intents (push-to-talk, wake word or spoken confirmation), or the confidence gate, and measure it (**N**, 3–5 days).

**B-M8. The acoustic experiment is in-distribution, and its statistics treat repeated measures as independent.**
- *Location:* B:L446–460, L604–623; Table 3; Fig. 4.
- *Why it matters:*
  - **In-distribution noise.** Evaluation noise is the last 30% of the same DREGON flight whose first 70% was used for training augmentation, and the parser was trained on recogniser output at 20, 10 and 5 dB. So the noise is in-distribution in type and largely in level.
  - **Unrealistic mixing.** Digital mixing onto quiet-room speech omits the Lombard effect (Ma et al., 2019) and does not reflect drone-audition SNRs (Mukhutdinov et al., 2023; Gulli et al., 2025). The intended operator–drone geometry is never stated.
  - **Invalid ANOVA.** A one-way ANOVA on arcsine-transformed binary outcomes over five conditions that reuse the same 200 utterances is not valid (B:L455–456, L615–616), and the repository's own analysis note says so (`results/exp3_pi_analysis.md`).
  - **Order-dependent causes.** The fixed-order cause partition (B:L605–609) makes the decomposition depend on the order chosen.
- *Remedy:*
  - State the acoustic scenario (distance, number of drones, microphone) and derive the SNR range from it.
  - Add a held-out noise recording (another flight or another dataset).
  - Keep Cochran's Q and McNemar, and drop the ANOVA and Tukey.
  - Present failure causes as an overlap table.
- *Tag:* text and statistics **E**. Held-out noise **N**, 1–2 days.

**B-M9. Recovery and parse-path misses have obvious fixes that are listed only as future work.**
- *Location:* B:L517–525, L546–547, L710–724, L780–784.
- *Why it matters:*
  - **Recovery.** It takes 1,195 ms at p95 against 300 ms, and the cause is identified as the server's one-second polling.
  - **Prefill.** The 751 ms prefill is uncached, while the companion's cached figure is 328.8 ms. Only a parity check stands between the two.
  - **Speech recognition.** It exceeds its allowance by 249 ms. Streaming or lighter recognisers exist (Moonshine, Distil-Whisper).

  A reviewer will ask why these were not done.
- *Remedy:*
  - Implement explicit slot cancellation or an in-process abort (design in `spikes/reports/S7_preemption_capability.md`) and re-measure (**N**, 2–4 days).
  - Run the prefix-cache parity check (`eval/check_parity_gguf.py`, `results/gate3_parity.md`) and re-time (**N**, about 1 day).
  - Benchmark Moonshine and Distil-Whisper on a held-out recording set, not the golden set (**N**, 2–3 days).

**B-M10. No operator, one speaker, and budgets not grounded in human factors.**
- *Location:* B:L108–114, L373–381, L754–757.
- *Why it matters:*
  - The motivation is a hands-free, eyes-up operator, but no operator used the system.
  - All audio is replayed recordings of the first author, a non-native speaker whose clean word error rate is 23.3%.
  - The 150, 300 and 2,500 ms budgets are asserted rather than derived.
  - Competing systems report user studies (FlockGPT) or real flights (SwarmGPT; Silva and Burke).
- *Remedy:*
  - Justify the budgets against reaction-time data (Woods et al., 2015).
  - Bring the accent analysis into the paper (**E**; `results/table16_asr_speaker_sensitivity.md`).
  - Add multi-speaker recordings (**N**, 2–3 weeks).
  - Ideally run a small within-subject study with 8–12 participants comparing voice with a baseline interface (**N**, 3–6 weeks including ethics approval).
  - At minimum, add a structured threats-to-validity section.

## B.5 Minor concerns

1. **"End of speech" vs "after endpointing".** The Fig. 2 caption and its x-axis (B:L509, "from the end of speech") contradict Table 2 and the text ("after endpointing", B:L493, L517). Align them (**E**; `eval/plots.py`).
2. **"The board added only 6–18 ms" (B:L80, L94, L775).** This reads as the board's whole share. The spotter also runs on the board. Say "after detection, compute and publication took 6–18 ms". **E.**
3. **Endpointing "met" (Table 2, B:L488).** The 480 ms is a configured constant (p50 = p95 = max), so a "met" verdict is vacuous. Mark it as configured. **E.**
4. **Loaded condition (B:L434–436).** It is a stress condition, since every trigger finds a decode. Add a "trigger during transcription" condition: the 74.2 ms worst frame leaves 6 ms of headroom (**N**, 1–2 days).
5. **Run order (B:L539).** The loaded run preceded the idle run, so the board was hotter at the start (about 73 °C against 52–58 °C). Report this. **E** (`results/exp2_raw/`).
6. **Requirements table (Table 4).** "Flight state machine", "Two simulation backends" and "Per-stage attribution" are engineering checks, not results. Move them to a separate list so that "seven met" is not read as performance. **E.**
7. **State-machine defect (B:L765).** The defect where legality is judged before validation is buried in Limitations. Describe the guard in `swarm/link.py` in Methods. **E.**
8. **Overloaded words.** "Slot" means formation slot and server slot (B:L408, L546, L712). "Offline" means offline scoring and offline operation (B:L544, Table 4). Disambiguate. **E.**
9. **Dependence on A for definitions.** CRR and the canonical comparison are defined by reference to the companion (B:L450–451), and so is the safe-failure target (Table 4 caption). Define them inline in two sentences. **E.**
10. **MIRA comparison (B:L752).** It uses different anchors, speakers and hardware. Keep the caveat next to the numbers. **E.**
11. **Speech-components paragraph (B:L389–401).** It sits inside §3.5 "Latency budget". Give it its own subsection, or move it to the supplement (`results/wake_training.md`). **E.**
12. **Figures.**
    - Fig. 4 duplicates Table 3; cut it.
    - Fig. 2(b) restates two table rows.
    - Move Fig. 3, Fig. 5 and Fig. 6 to the supplement.
    - Fig. 5 uses red and green with no legend.
    - In Table 1 and Table 4, narrow columns wrap badly; collapse the five "Parse only" rows of Table 1 into one.
    
    **E.**
13. **Abstract (B:L70–87).** About 260 words, about 30 numbers, and eleven result sentences with no ranking or implication. Rewrite to about 200 words. Drop the formation sentence, which also has a dangling modifier ("Commanded directly…"). **E.**
14. **Highlight 4 (B:L93).** It is cryptic. Give the number: "Reflex p95 is 545 ms against 150 ms; post-detection compute is 6–18 ms". **E.**
15. **Contribution framing (B:L127–158).** "The contribution we claim is narrow", followed by "These are engineering, not claimed as novel", reads as "no contribution". State the claims positively and verifiably. The pronoun "them" (B:L151) has no clear referent. **E.**
16. **Power draw** of the Pi is not measured, but it is relevant to a carried operator device (**N**, 0.5 day). Also mention that the uncooled board throttles in 100% of trials (`results/thermal_headroom.md`) as a deployment constraint. **E.**
17. **Placeholders and front matter.** The same placeholders, `[draft]` markup (B:L21), data-availability issues and generative-AI wording as A minor items 17–19. Fig. 5 is "distributed as rendered" (B:L800); add its script. **E.**
18. **ASR prompt provenance.** State that the Whisper initial prompt was fixed on spike S6 data before the golden set was recorded. A reviewer will ask about leakage (**E**; `spikes/reports/S6_asr_accent_check.md`, `data/asr.py`).

## B.6 What to add, remove and restructure

**Add**
- A hazard analysis and a formal statement of the membership rule; runtime-assurance positioning; ISO 13850 framing (B-M1).
- A delay–threshold curve and a re-trained, re-measured spotter (B-M2).
- Single-path, voice-onset-hold and regex baselines on identical audio (B-M3).
- The voice → vehicle run on PyFlyt with a live microphone, Wi-Fi with loss, and networking disabled; stopping distance and time to rest (B-M4).
- Reflex error rates in noise, a longer ambient stream, and more speakers (B-M5, B-M10).
- The out-of-domain false-command rate and a gate for movement intents (B-M7).
- Slot cancellation, prefix-cache parity and an alternative ASR (B-M9).
- 2025–2026 related work (B.7).

**Remove**
- "Formation control" from the title and abstract unless the voice → formation run exists (B-M6).
- The formation ANOVA and Tukey, the CRR ANOVA and Tukey, and the "collisions met" row (B-M6, B-M8).
- Fig. 4, Fig. 2(b), and the repeated narration of Results in the Discussion.
- Engineering checklist rows from the requirements table (minor 6).
- Duplicated system description and numbers shared with A (X-1).

**Restructure**
- Lead with the rule and the budget, with the runtime as the test bed. Results order: (1) reflex path against baselines, (2) closed-loop stop on vehicles, (3) parse path, (4) robustness, (5) supplementary formation check.
- Move speech-component details and keyword-spotter training to Methods or the supplement.
- Rewrite Limitations as a structured threats-to-validity list.
- Title options:
  - "Which voice commands may bypass interpretation? A reflex path with a safety-derived membership rule and per-stage latency budget for drone swarms on a CPU-only edge device"
  - "Skipping the language model for stop commands: a keyword-spotter reflex path for voice-controlled drone swarms on a Raspberry Pi 5"

## B.7 Missing references

Each was resolved during this review. None is in `articles/ingenieur/references.bib`. References 1–2 of A.7 (Ahmed et al., 2026; Nguyen et al., 2026) should also be cited in B at B:L168–185.

**Language-model swarm and drone control**
1. Schuck, M., Dahanaggamaarachchi, D.O., Sprenger, B., Vyas, V., Zhou, S., Schoellig, A.P., 2025. SwarmGPT: Combining Large Language Models with Safe Motion Planning for Drone Swarm Choreography. IEEE Robotics and Automation Letters (accepted, per arXiv). https://arxiv.org/abs/2412.08428. **Cite at** B:L177–185. It pairs an LLM with a safety filter, flown on real drones.
2. Lykov, A., Karaf, S., Martynov, M., Serpiva, V., Fedoseev, A., Konenkov, M., Tsetserukou, D., 2024. FlockGPT: Guiding UAV Flocking with Linguistic Orchestration. arXiv:2405.05872. https://arxiv.org/abs/2405.05872. **Cite at** B:L177–185 and B:L200–202. It includes a user study.
3. Li, P., Zhou, L., 2025. LLM-Flock: LLM-Driven Decentralized Multi-Robot Flocking via Influence-Based Consensus with Collision Avoidance. arXiv:2505.06513. https://arxiv.org/abs/2505.06513. **Cite at** B:L200–202.
4. Zhang, Z., Deng, X., Qu, G., Nakahira, Y., 2026. UAVGENT: A Language-Guided Distributed Control Framework. arXiv:2602.13212. https://arxiv.org/abs/2602.13212. **Cite at** B:L745–753. It is an example of a formal guarantee.
5. Choutri, K., Fadloun, S., Khettabi, A., Lagha, M., Meshoul, S., Fareh, R., 2025. Leveraging Large Language Models for Real-Time UAV Control. Electronics 14(21), 4312. https://doi.org/10.3390/electronics14214312. **Cite at** B:L172–177 and B:L745–753. It covers English and Arabic voice control with offline recognition.
6. Azzouni, A., Pujolle, G., 2025. Autonomous Drone Swarms Using Lightweight LLMs. Proceedings of the 1st GENZERO Workshop, Springer. https://doi.org/10.1007/978-981-95-1050-4_3. **Cite at** B:L183–185, where "none of these systems" is claimed.
7. Hustermeier, F., Yigitbas, E., 2025. A Novel Approach for Controlling Drone Swarms: Integrating LLMs and Augmented Reality. INTERACT 2025, LNCS. https://doi.org/10.1007/978-3-032-04999-5_27. **Cite at** B:L177–185.
8. De Cubber, G., Doroftei, D., 2026. Human-Friendly Control of Drones and Drone Swarms Using Natural Language and AI-Based Task Decomposition. AHFE International. https://doi.org/10.54941/ahfe1007683. **Cite at** B:L168–185 and in the human-factors limitation.
9. Casado Faulí, A.M., Malizia, M., Hasselmann, K., Le Flécher, E., De Cubber, G., Lauwens, B., 2024. HADRON: Human-friendly Control and Artificial Intelligence for Military Drone Operations. arXiv:2408.07063. https://arxiv.org/abs/2408.07063. **Cite at** B:L168–185.
10. Jin, Y., et al., 2026. HoverAI: An Embodied Aerial Agent for Natural Human-Drone Interaction. arXiv:2601.13801. https://arxiv.org/abs/2601.13801. Optional, at B:L170–177.
11. Liang, J., Huang, W., Xia, F., Xu, P., Hausman, K., Ichter, B., Florence, P., Zeng, A., 2023. Code as Policies: Language Model Programs for Embodied Control. ICRA 2023. https://arxiv.org/abs/2209.07753. **Cite at** B:L111 beside SayCan.

**Safety, runtime assurance and emergency stop**

12. Ravichandran, Z., Robey, A., Kumar, V., Pappas, G.J., Hassani, H., 2025. Safety Guardrails for LLM-Enabled Robots. arXiv:2503.07885. https://arxiv.org/abs/2503.07885. **Cite at** §3.4 (B:L354–371).
13. Mehmood, U., Sheikhi, S., Bak, S., Smolka, S.A., Stoller, S.D., 2022. The Black-Box Simplex Architecture for Runtime Assurance of Autonomous CPS. NASA Formal Methods, LNCS, 231–250. https://doi.org/10.1007/978-3-031-06773-0_12. **Cite at** §3.1–3.3.
14. ASTM F3269-21. Standard Practice for Methods to Safely Bound Behavior of Aircraft Systems Containing Complex Functions Using Run-Time Assurance. https://store.astm.org/f3269-21.html. **Cite at** §3.3.
15. ISO 13850:2015. Safety of machinery — Emergency stop function — Principles for design. https://www.iso.org/standard/59970.html. The catalogue entry was confirmed via national-body listings, for example https://scc-ccn.ca/standardsdb/standards/8157956. **Cite at** §3.3 to state that a voice stop supplements a hardware stop.
16. Leveson, N.G., 2012. Engineering a Safer World: Systems Thinking Applied to Safety. MIT Press. https://doi.org/10.7551/mitpress/8179.001.0001. **Cite at** the hazard analysis of B-M1 (STPA).
17. Davila, A., Colan, J., Hasegawa, Y., 2024. Voice control interface for surgical robot assistants. MHS 2024 (IEEE). https://doi.org/10.1109/MHS63891.2024.10856297. **Cite at** B:L187–195 as prior art for a voice "stop" command.

**Fast and slow paths**

18. Chen, H., Liu, J., Gu, C., Liu, Z., Zhang, R., et al., 2025. Fast-in-Slow: A Dual-System Foundation Model Unifying Fast Manipulation within Slow Reasoning. arXiv:2506.01953. https://arxiv.org/abs/2506.01953. **Cite at** B:L187–195.
19. Lee, J., Shin, H., Ko, J., 2026. IROS: A Dual-Process Architecture for Real-Time VLM-Based Indoor Navigation. arXiv:2601.21506. https://arxiv.org/abs/2601.21506. **Cite at** B:L187–195.
20. Zhu, X., Meng, Q., Yu, L., Zhang, W., Ma, Z., et al., 2026. FSD-VLN: Fast-Slow Dual-System Modeling for Aerial Long-Horizon Vision-Language Navigation. arXiv:2607.08359. https://arxiv.org/abs/2607.08359. **Cite at** B:L187–195. None of 18–20 restricts the fast path by a safety rule; say so to sharpen the contribution.

**Speech, keyword-spotting latency and drone audition**

21. Jose, C., Wang, J., Strimel, G.P., Khursheed, M.O., Mishchenko, Y., Kulis, B., 2022. Latency Control for Keyword Spotting. Interspeech 2022. https://doi.org/10.21437/Interspeech.2022-10608. **Cite at** B:L705–710 (B-M2).
22. Jeffries, N., King, E., Kudlur, M., Nicholson, G., Wang, J., Warden, P., 2024. Moonshine: Speech Recognition for Live Transcription and Voice Commands. arXiv:2410.15608. https://arxiv.org/abs/2410.15608. **Cite at** B:L721–724.
23. Gandhi, S., von Platen, P., Rush, A.M., 2023. Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling. arXiv:2311.00430. https://arxiv.org/abs/2311.00430. **Cite at** B:L721–724.
24. Mukhutdinov, D., Alex, A., Cavallaro, A., Wang, L., 2023. Deep Learning Models for Single-Channel Speech Enhancement on Drones. IEEE Access 11, 22993–23007. https://doi.org/10.1109/ACCESS.2023.3253719. **Cite at** B:L446–448 (B-M8).
25. Gulli, A., Fontana, F., Drioli, C., Salvati, D., Ferrin, G., 2025. Enhancing drone audition with rotor-conditioned deep models. EURASIP Journal on Audio, Speech, and Music Processing. https://doi.org/10.1186/s13636-025-00425-2. **Cite at** B:L446–448.
26. Ma, P., Petridis, S., Pantic, M., 2019. Investigating the Lombard Effect Influence on End-to-End Audio-Visual Speech Recognition. Interspeech 2019. https://arxiv.org/abs/1906.02112. **Cite at** B:L446–448 and in Limitations.
27. Woods, D.L., Wyma, J.M., Yund, E.W., Herron, T.J., Reed, B., 2015. Factors influencing the latency of simple reaction time. Frontiers in Human Neuroscience 9, 131. https://doi.org/10.3389/fnhum.2015.00131. **Cite at** B:L373–381 to ground the budgets.

**Statistics.** Use Hanley and Lippman-Hand (1983; A.7 #35) for the zero-numerator bounds of B-M5, and Eliasziw and Donner (1991; A.7 #32) for the clustered paired tests.

## B.8 Suggested venues

| Rank | Journal | Quartile / IF (secondary) | APC (hybrid) | Why it fits | What B must change |
|---|---|---|---|---|---|
| 1 | **Journal of Systems Architecture** (Elsevier) | JCR Q1 (Hardware & Architecture), IF ≈ 5.3 | US$2,260 | Real-time embedded systems; the paper's substance is a runtime architecture, a preemption protocol and a latency budget measured on a board. Scope says case studies "must contribute more than just a design for a particular piece of software", so B-M1 and B-M3 are essential. | Lead with the architecture and the rule. Make formation supplementary. Add baselines and the closed-loop run. |
| 2 | **Robotics and Autonomous Systems** (Elsevier) | JCR Q1, IF ≈ 6.0 | US$2,940 | Topic fit (autonomous systems, human–robot command). | Only with the closed loop (B-M4) and ideally SITL or real vehicles. Simulation-only work is high risk here. Slowest review times of the three. |
| 3 | **Internet of Things** (Elsevier) | JCR Q1 (Information Systems), IF ≈ 7.1–7.6 (sources conflict) | US$2,800 | Edge-device and human-in-the-loop framing. | Thin fit unless the Wi-Fi hop and the device–workstation split are measured (B-M4). |
| Alt. | Journal of Intelligent & Robotic Systems (Springer) | JCR Q3, IF ≈ 2.8 | not confirmed | Accepts simulation-heavy robotics work. | **Not Q1.** Fallback only. |

---

# Cross-manuscript section

## X-1. Overlap and salami-slicing risk

**Measured overlap**
- **Verbatim prose.** Overlap is negligible: under 1% of 8-grams, all boilerplate. A similarity checker will not flag the prose.
- **Data and measurements.**
  - Both papers use the same 200-item golden set, the same board and cooler, the same model and grammar, and the same DREGON flight.
  - About 100 numbers repeat, including 0.935 EM, 3,122 and 3,602 ms, 751 and 328.8 ms prefill, 23.3% WER, and CRR 0.690 and 0.590.
  - 14 of 40 references are shared, and the opening related-work paragraph of each cites the same five systems for the same purpose.
- **Circular dependence.** Each paper takes headline numbers from the other, and both cite the other as "in preparation":
  - A's selection rule and abstract rest on B's end-to-end measurement (A:L85–88, L611–619).
  - A's limitations use B's CRR (A:L692).
  - B takes its model, grammar, EM, canonical comparison and safe-failure target from A.

**Assessment.** The research questions are genuinely different (model selection against interface architecture), so this is not duplicate publication. The editorial risk is the circular dependence on unverifiable companions, plus the perception that each paper is thin on its own. If both reach the same editor, a merge request is likely.

**How to make each stand alone**
1. **Give each number one home.**
   - A owns model-level accuracy, quantisation, grammar, thermal behaviour and language-model-stage timing.
   - B owns the reflex path, the full-chain budget, noise robustness, preemption and the membership rule.
   - Remove 3,122/3,602 ms and the CRR figures from A's abstract and results; keep one Discussion sentence citing B.
2. **Redefine A's selection rule** on language-model-stage constraints A measures itself (A-M7). This also removes the circularity.
3. **Make B self-contained.** Define CRR, the canonical comparison, the safe-failure target and the parser's EM (with its test set) in two or three sentences.
4. **Remove duplicate content.**
   - Keep the "language models as robot command interfaces" paragraph in B and shorten it in A.
   - The abstention argument belongs to A; B keeps one sentence.
   - Do not draw two stacked-latency figures.
5. **Make the companion verifiable.** Post both as preprints with DOIs before submission (arXiv, or Zenodo if arXiv is unsuitable). Cite the preprint, upload the companion as a "related manuscript" file, and state in each cover letter exactly what is shared and what is exclusive. Elsevier permits preprints at any time ("Sharing" policy, https://www.elsevier.com/about/policies-and-standards/sharing); check the target journal's anonymisation rules first.
6. **Submit to different journals,** A first. B's model comes from A, so A should be in the system first.

## X-2. Split or merge (decision for the authors)

The reviewers disagreed here. The adversarial reviewer favoured merging; the editorial reviewer and the venue scout favoured two papers in two journals. My resolution:

- **Two papers (recommended)**, but only if B gets its new evidence: the closed-loop run (B-M4), a re-measured reflex path (B-M2), baselines (B-M3) and the hazard analysis (B-M1). A is viable alone after the cheap items in its plan.
- **One merged paper** is the fallback if B's new work cannot be done within your timeline. A becomes the model-selection section, and B's reflex path, rule and budget become a system section, in a single submission to *Journal of Systems Architecture*. This removes the circular citation and the salami perception. It also keeps B's honest negative findings, which are more credible as one part of a larger study than as a paper's whole result.

## X-3. Venue pairing

- A goes to **EAAI**, reformatted to single column, with the AI contribution separated in the abstract and the dataset archived with a DOI. Fallback: Computers and Electrical Engineering.
- B goes to **Journal of Systems Architecture**. Fallback: Robotics and Autonomous Systems, only with closed-loop or vehicle evidence.
- Different journals keep the editorial and reviewer pools apart.

## X-4. Prioritised action plan

Impact means the effect on acceptance chances: ★★★ decisive, ★★ substantial, ★ helpful. Effort is working time.

### Must do before submission

| ID | Manuscript | Action | E/N | Effort | Impact |
|---|---|---|---|---|---|
| P1 | Both | Fill the corresponding-author, e-mail and ORCID placeholders. Switch `changes` to `[final]` and strip the audit macros. Add a licence, archive a release with a DOI, and fix the repository URL. Add data-licence and voice-consent lines. | E | 0.5 day | ★★★ (desk-reject risk) |
| P2 | Both | Post both preprints with DOIs. Decouple the numbers (X-1, items 1–4). Upload the companion as a related manuscript and disclose the relationship in each cover letter. | E | 1–2 days | ★★★ |
| P3 | A | Family-clustered statistics, a per-family table, EM with and without F106, and "item-level" wording (A-M1). | E | 3 h | ★★★ |
| P4 | A | Baselines: few-shot + grammar and a rule-based parser at minimum; then an encoder and an upper-bound model (A-M2). | N | 2 days minimum; 4–7 days full | ★★★ |
| P5 | A | Correct the provenance of the confirmatory families and the primary split. Report the contrast on all three splits. Rewrite "at least as accurate" (A-M5). | E | 2 h | ★★★ (integrity) |
| P6 | A | Reposition the contribution. Cite Ahmed 2026, Nguyen 2026, SLOT, Chavan, Ray, Kavathekar and TinyAgent. State the search protocol (A-M3). | E | 1 day | ★★★ |
| P7 | A | Redefine the selection rule on language-model-stage constraints. Measure SmolLM2 end to end (A-M7). | E + N | 0.5 day + 3 h | ★★ |
| P8 | A | Demote the family control from the abstract, highlights and contributions (A-M10). | E | 1 h | ★★ |
| P9 | A | Report the clockwise/counter-clockwise training counts as the cause of the yaw-sign errors (A-M4a). | E | 2 h | ★★ |
| P10 | A | Out-of-domain provenance and per-source false-command rates (A-M8i). Per-model grammar ablation with a rule-of-three bound (A-M6iv). | E | 0.5 day | ★★ |
| P11 | B | Run the existing voice → state machine → controller chain on PyFlyt with a live microphone and Wi-Fi. Measure stopping distance and time to rest. Run with networking disabled (B-M4). | E code + N runs | 1–3 days | ★★★ |
| P12 | B | Single-path baseline: the 40 stop takes through the parse path. Add a voice-onset hold (B-M3a–b). | N | 0.5–3 days | ★★★ |
| P13 | B | Hazard analysis and a formal statement of the membership rule, including whether abort belongs on the reflex path. Runtime-assurance and ISO 13850 positioning (B-M1 i–iv). | N | 3–5 days | ★★★ |
| P14 | B | Correct the clamp interpretation. Mark collisions "zero by construction". Drop "formation control" from title and abstract, or replace it with P11. Run a clamp-off ablation (B-M6). | E + N | 0.5 day | ★★★ (factual error) |
| P15 | B | Delay–threshold curve and forced-alignment anchor. Then shorten the spotter delay and re-measure against 150 ms, or justify a revised budget (B-M2). | N | hours (curve); 1–2 weeks (retrain) | ★★★ |
| P16 | B | Take as the unit (n = 39) with exact intervals. Report the out-of-domain false-command rate (B-M5, B-M7). | E | 0.5 day | ★★ |
| P17 | Both | Drop the ANOVA and Tukey analyses (CRR, formation). Use exact McNemar throughout. Add order-statistic CIs for percentiles (A-M7, B-M8, A minor 5). | E | 0.5 day | ★★ |
| P18 | Both | Rewrite abstracts to about 200 words with an implication sentence; contributions as verifiable claims; replace weak highlights; consolidate figures and tables. For EAAI: single column and no undefined acronyms in title or abstract. | E | 2–3 days | ★★ |

### Would strengthen

| ID | Manuscript | Action | E/N | Effort | Impact |
|---|---|---|---|---|---|
| S1 | A | Fix rotation data or schema (balanced clockwise examples, or a `cw`/`ccw` enumeration), retrain, and show the effect on EM and the ranking (A-M4b). | N | 1–2 days | ★★★ |
| S2 | A | Quantisation ladder (GGUF f16, Q6_K to Q2_K, Q4_0, IQ4_XS) with an equivalence margin and flip rates (A-M6 i–iii). | N | 1–2 days | ★★ |
| S3 | A | Grammar ablation where it can matter (ASR transcripts, few-shot models, low bits, T > 0). Grammar overhead timed on the Pi (A-M6 v–vi). | N | 1.5 days | ★★ |
| S4 | A | Score all six artefacts on the existing Whisper transcripts (A-M9i). | E + N | hours | ★★ |
| S5 | A | Confidence gate with a risk–coverage curve (A-M8ii). | N | 2–3 days | ★★ |
| S6 | A | Add Qwen3-0.6B and Gemma-3-270M/1B, three seeds per model, and a within-family size ladder (A-M9 ii–iii). | N | 1–2 weeks | ★★ |
| S7 | A | 200 timing trials, Q8_0 timed, deployed uncached timing, thermal replicate with clock logging (A-M7). | N | 1–2 days | ★★ |
| S8 | A | Power and energy per command (A minor 12). | N | 1 day | ★ |
| S9 | B | Reflex error rates under rotor noise; ambient stream ≥ 10 h; investigate the live false hover (B-M5). | N | 2–4 days | ★★ |
| S10 | B | Slot cancellation for recovery; prefix-cache parity; trigger-during-transcription condition (B-M9, B minor 4). | N | 3–6 days | ★★ |
| S11 | B | Regex and grammar-ASR baselines on the golden set; Moonshine or Distil-Whisper on a held-out recording set (B-M3c–d, B-M9). | N | 3–5 days | ★★ |
| S12 | B | Gate for movement intents (push-to-talk, confirmation or confidence) (B-M7). | N | 3–5 days | ★★ |
| S13 | B | Datagram-loss injection over Wi-Fi with acknowledgement or repetition for stops (B-M4). | N | 1–2 days | ★ |
| S14 | B | Model-check the ordering rule and the state machine (B-M1v). | N | 3–5 days | ★ |
| S15 | Both | Multi-speaker recordings: golden patterns and keyword takes, ≥ 5 speakers (A-M9iv, B-M10). | N | 2–3 weeks | ★★ |
| S16 | B | Small operator study (8–12 participants) or flights on real or SITL vehicles (B-M10). | N | 3–6 weeks | ★★ (★★★ for RAS) |
| S17 | Both | Graphical abstracts (A minor 20). | E | 2–3 h each | ★ |

---

## Appendix: claims dropped or corrected during consolidation

- **Vocabulary-size effect on grammar cost, and Llama-3 digit grouping.** Raised by the domain reviewer from memory; not verified. They are kept only as "may" in A-M6, and the remedy (time the grammar) settles the point either way.
- **llama.cpp ARM Q4_0 repacking speed.** Not verified; dropped as a claim. Q4_0 is kept in the quantisation ladder.
- **"39 negative against 81 positive training rotate labels".** Superseded by the direct count (F2): 6 of 120 contain "clockwise".
- **The 10,086 ms parse latency from speech onset (B domain reviewer).** Verified in `results/exp2_latency_budget.md`, but it is for long golden-set commands, not stop phrases, so it is not used as a stop-latency comparison. B-M3a asks for the like-for-like measurement instead.
- **Choutri et al. (2025) uses "cloud Gemini".** One reviewer reported this, but the full text could not be opened. The reference is kept; that detail is not.
- **Journal metrics.** Impact factors, quartiles and review times are secondary-source values because ScienceDirect and Scimago block automated access. Only the APCs (Elsevier price list of 05-Oct-2026), the EAAI desk-reject rules and the ESWA defence exclusion were read on Elsevier's own pages.
