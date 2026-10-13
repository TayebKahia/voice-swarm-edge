# Verification review: Master and Ingénieur journal articles

Date: 2026-10-13. Scope: correctness only (numbers, claims, honesty, labels, cross-article
consistency, citations, section discipline, format limits). No prose or journal-fit review;
that follows the supervisor's feedback. The articles were not edited.

## How this review was done

Four independent subagents ran in parallel, as specified:

| # | Scope | Model / effort | Items | Result |
|---|---|---|---|---|
| 1 | Every number in `articles/master/main.tex` | Sonnet 5.5 / high | 233 | 229 verified, 0 mismatch, 3 unsure |
| 2 | Every number in `articles/ingenieur/main.tex` | Sonnet 5.5 / high | 199 | 191 verified, 5 mismatch, 3 unsure |
| 3 | Claims, honesty, labels, consistency, Results/Discussion, limits | Opus 5.5 / high | — | 4 proposed errors, 15 should-fix, minors |
| 4 | Citations and bib entries, arXiv status | Sonnet 5.5 / high | 114 cite rows | 108 verified, 4 mismatch, 2 unsure |

Timing figures were recomputed from raw rows (`results/exp1_cooled.csv`, `results/exp1.csv`,
`results/exp2_raw/*.jsonl`), formation statistics from `results/exp4.csv`, CRR/McNemar/Cochran
from `results/exp3_pi_preds`, keyword corpus counts from `data/wake/wake_manifest.json`, and
figure labels from the figure PDFs and `eval/plots.py`.

The coordinator then re-checked **every** MISMATCH and UNSURE item, and every finding of
subagent 3, against the sources before it entered this report. **Dropped as false alarms or
unconfirmed: 7.**

1. M "about 35% or more below the ceiling" (1 − 1.631/2.5 = 34.8%; "about 35%" holds).
2. M TypeFly "below 1.5 s": the figure is in the thesis (`thesis/ingenieur/ch3_state_of_the_art.tex:73`) and in the paper body.
3. M LLM2Swarm "TinyLlama on a Raspberry Pi 5": the figure is in the thesis (`…state_of_the_art.tex:255,284`).
4. I "Two stages explain the 622 ms excess" (l. 673): a qualitative statement; the article itself warns that percentiles do not add.
5. I cancelled request "within 6.5 ms" (l. 505), proposed as an error: `results/exp2.csv:23` gives p95 = 6.5. The thesis and `exp2_analysis.md` round the same value to 6.
6. I "24 Hz effective control rate" (l. 710): the figure is in the thesis (`ch4_implementation.tex:438–440`).
7. M "60 timed trials instead of 200 planned": I could not confirm a 200-trial plan in the thesis.

Only confirmed findings follow. Severity is the coordinator's: **error**, meaning the text is
false or contradicts its source; **should-fix**, meaning it misleads, overstates or breaks an
agreed rule; and **minor**.

---

## Verdicts by check

| Check | Master | Ingénieur |
|---|---|---|
| 1. Numbers trace to sources | Pass. 0 errors; 1 minor rounding inconsistency (M-14) | **3 errors** (I-1, I-2, I-3); 1 should-fix (I-10) |
| 2. No claim of undone work | Pass with 1 minor (M-15) | Pass on the six known facts; 3 should-fix (I-4, I-5, I-6), 2 minor |
| 3. Misses reported honestly | 5 should-fix (M-1 to M-5) | 5 should-fix (I-7 to I-11), 2 minor |
| 4. No internal labels | 1 should-fix (M-6) | Pass |
| 5. Cross-article consistency | See cross-article section: 2 should-fix, 1 minor; no duplicate-publication overlap | (same) |
| 6. Citations | 2 should-fix, 2 minor | 2 should-fix, 3 minor |
| 7. Results vs Discussion | 4 minor | 4 minor |
| 8. Limits | Pass: abstract 237–241 words; 5 highlights of 69–82 chars; 6 keywords | Pass, at the limit: abstract **250** words (coordinator recount: 250); 5 highlights of 72–81 chars; 6 keywords |

---

## Master article (`articles/master/main.tex`)

### Should-fix

**M-1 — Cached-prefill miss of the selected model is never stated** (l. 521–523, 602–603; check 3)
- Problem: Qwen2.5-0.5B's prefill p95 is 328.8 ms against the 250 ms allowance. The text gives
  the figure and "8 of 60 trials above 250 ms", but states a miss only for Llama. The list at
  l. 602–603 ("meets the EM, intent-F1, decode, combined …") silently omits prefill.
- Source: `thesis/master/ch4_results.tex:130` says the 250 ms allowance is "the one language-model stage
  allowance the selected configuration misses".
- Fix: add "and misses the 250 ms prefill allowance" to l. 521, and "it misses only the prefill
  allowance" to l. 602–603.

**M-2 — "Combined allowance met" and "278 ms to spare" hold only with the prompt cache** (l. 523–524, 602; check 3)
- Problem: the deployed parser runs without the cache. Its combined prefill-plus-decode p95 is
  1,658 ms, which misses 1,350 ms. The article gives the uncached prefill (751 ms) one sentence
  later but never the uncached combined verdict.
- Source: `results/exp2_analysis.md:16` (combined row 1,112 / 1,658, verdict MISSES).
- Fix: "With a cached prefix, Qwen2.5-0.5B has 278 ms to spare; as deployed, without the cache,
  its combined p95 is 1,658 ms and misses the allowance [companion]". Qualify l. 602 the same way.

**M-3 — "Neither constraint … is satisfied by any configuration" overstates** (l. 588–589; check 3)
- Problem: full-stack memory was not measured, and SmolLM2 was not measured end to end, so
  neither satisfaction nor failure is shown. The next sentences say so, which contradicts l. 588.
- Source: `thesis/master/ch4_results.tex:339,352,354` ("no configuration is shown to satisfy the
  ceiling; none is shown to miss it either").
- Fix: "No configuration is shown to satisfy both constraints."

**M-4 — Highlight 2 and the conclusion turn a non-significant difference into equivalence** (l. 79, 689; check 3)
- Problem: "A 0.5B model matches a 1.2B model" and "as accurate as a model 2.5 times its size"
  rest on 5 discordant items (p = 0.0625). Absence of significance is not equivalence, and the
  article itself says no split of five items could reach α.
- Source: `thesis/master/ch5_discussion.tex:254–256`.
- Fix, highlight (79 chars): "A 0.5B model scores 0.935 against 0.910 for a 1.2B model, a
  non-significant gap". Conclusion: "at least as accurate on this test set as a model 2.5 times its
  size (difference not significant)".

**M-5 — Highlight 5 and the fourth contribution generalise one model's result** (l. 82, 135–136; check 3)
- Problem: only Qwen2.5-0.5B crosses the floor with cooling. SmolLM2 stays above 20 tokens/s
  uncooled (24.11), and Llama is below it even when cooled (14.53).
- Source: `results/thermal_headroom.md:12,14`; article Table 1.
- Fix, highlight (80 chars): "Qwen2.5-0.5B falls below the 20 tokens/s throughput floor without
  active cooling".

**M-6 — Results file name and thesis table number in a caption** (l. 365; check 4)
- Problem: "Full table: `results/table17_model_comparison.md`" leaks a repository path that
  carries the thesis table number "table17".
- Fix: delete the sentence (the Data availability section already covers the repository), or
  write "The full table is in the repository (Data availability)."

**M-7 — GPTQ cited for an on-device motivation it does not state** (l. 162–163; check 6)
- Problem: GPTQ motivates itself by compute and memory cost (running a 175B model on one GPU);
  only AWQ names on-device deployment.
- Source: the thesis splits the two, `thesis/master/ch3_state_of_the_art.tex:82` ("footprint [gptq] … on-device deployment is one of its stated motivations [awq]").
- Fix: "GPTQ reduces the memory footprint of large models~\citep{gptq}, and AWQ lists on-device
  deployment among its motivations~\citep{awq}."

**M-8 — A figure is cited to the companion paper that the companion does not report** (l. 593–594; check 6, also check 5)
- Problem: "The shortest speech-recognition time in that experiment is 1,063 ms [companion]".
  The Ingénieur article reports only p50 1,246 and p95 1,449. The value is correct from raw data
  (subagent 1: 2,282.45 + 1,063.38 = 3,345.8, which gives 3,346), but a reader cannot find it in
  the cited work.
- Source: present only in `thesis/master/ch4_results.tex:347`.
- Fix: either add the minimum recognition time to the Ingénieur latency text, or cite the data
  repository for it here.

**M-9 — `slmquant` has a published version** (bib; check 6)
- SLMQuant appeared at RichMediaGAI '25 (ACM MM 2025 workshop), DOI 10.1145/3746262.3761973. The
  coordinator confirmed this independently. Update the bib entry from arXiv-only.

### Minor

- **M-10** (check 6): `smollm2` is reported by subagent 4 as published at COLM 2025 (OpenReview
  3JiCl2A14H). The coordinator's search did not confirm it, so verify it on OpenReview before
  updating the entry.
- **M-11** (check 6): `geng2023` lacks `pages` (EMNLP 2023, pp. 10932–10952), and so does `qin2021`
  (IJCAI 2021, pp. 4577–4584). BibTeX warns on both.
- **M-12** (check 3): some verdicts are left for the reader to infer from the numbers. These are
  Llama's decode miss (1,842 ms against 1,100 ms, +67%), Llama's throughput miss (14.53 tokens/s),
  SmolLM2's EM miss (0.760 against 0.85), and the uncooled run's 100% throttled trials against the
  5% limit. A one-line verdict for each would make the reporting of misses complete. The fp16
  training of bf16-released checkpoints (`thesis/master/ch3_method.tex:349–354`) is also not mentioned.
- **M-13** (check 3): the abstract does not say that the selection needed the failure clause
  (re-baselining).
- **M-14** (check 1): SmolLM2's synthetic EM appears as 0.812 in Table 1 (l. 374–375) and as 0.8125
  at l. 501. Both match the repo tables, which round half-even (0.8125 to 0.812, and Danube3's 0.8875
  to 0.887). Use one form, or round half-up throughout (0.813 / 0.888).
- **M-15** (check 2): l. 122–123 "commands a swarm of five drones" should be "five simulated drones".
- **M-16** (check 7): interpretation inside Results:
  - l. 411–412, why overlapping intervals are compatible with a paired test;
  - l. 498–499, a test-power argument;
  - l. 500–502, "identical EM does not imply identical predictions".

  Move them to Discussion or cut them.
- **M-17** (check 7): the per-model validation gains (0.025, 0.038) first appear in Discussion
  (l. 617). Methods gives only the range 0.008–0.038.

---

## Ingénieur article (`articles/ingenieur/main.tex`)

### Errors

**I-1 — "Closest approach … in every one of the 150 trials was 0.800 m" is false** (l. 612–613, repeated at l. 694; check 1)
- Problem: 144 trials are at 0.800 m. Six trials are higher: trials 14 and 24 of each formation,
  at 0.894–0.895 m and 0.823–0.824 m. These are exactly the six trials with no clamp intervention.
  0.800 m is the minimum over all trials, not the per-trial value.
- Source: `results/exp4.csv` (`min_pair_distance`, rechecked by the coordinator);
  `results/exp4_formation.md:11` and `thesis/ingenieur/ch5_validation.tex:375` say only "the closest approach recorded".
- Fix, l. 612: "The closest approach was 0.800 m, the clamp distance, in each of the 144 trials in
  which the clamp intervened, and 0.82–0.89 m in the other six." Fix, l. 694: "and in every trial in
  which the clamp intervened, the closest approach equals the clamp distance."

**I-2 — "4.7 m … larger than every admissible inter-drone spacing (1–5 m)" is arithmetically false** (l. 301–302; check 1)
- Problem: 2.0 m/s × 2.350 s = 4.70 m, which is less than the 5.0 m maximum spacing.
- Source: `schema/schema.py:114` (`"spacing": (1.0, 5.0)`). The thesis has the same error at
  `thesis/ingenieur/ch3_architecture.tex:189–190`, so fix it there too.
- Fix: "comparable to the largest admissible inter-drone spacing (5 m)". The argument is unchanged,
  because a stop arriving 2.35 s late lets a vehicle cover almost a full spacing.

**I-3 — The keyword-spotter training set is overstated** (l. 358–361; check 1)
- Problem: "trained on 3,000 synthetic positive clips … and 5,040 negative clips" gives the whole
  corpora, including the validation and test splits. The 300 test takes per class that are scored at
  l. 400 and l. 525 are part of the 3,000, so the sentence as written implies training on the test
  set.
- Source: `data/wake/wake_manifest.json`. Positives: 1,500 per class, split train 1,050 / val 150 /
  test 300 (2,100 train in all). Negatives: 5,040 in all (1,936 LibriSpeech + 2,384 Speech Commands
  + 720 near-miss), split train 3,930 / val 348 / test 762.
- Fix: "The corpus holds 3,000 synthetic positive clips … and 5,040 negative clips …; the heads were
  trained on 2,100 positives and 3,930 negatives, with the rest held out for validation and test."

### Should-fix

**I-4 — The physical effect of hold and abort is asserted, but no code applies them to the controller** (l. 283–284, 308–310; check 2)
- Problem: Table 1 and §3.3 describe "stops lateral motion" and "descends in place". The controller
  accepts only formation commands, and no code maps `hover`, `abort` or `land` onto it, so neither
  effect was ever executed. The time to rest after a hold was not measured.
- Source: `thesis/ingenieur/ch4_implementation.tex:331–335`.
- Fix: in §3.3 or Limitations, state that these are the intended effects on the designed route,
  that the mapping to the controller is not built, and that time-to-rest was not measured.

**I-5 — The ordering rule is presented as a system guarantee, but no trial exercised it** (l. 114–115, 258–266; check 2)
- Problem: no cancelled decode completed, so the rule never had to discard a message. It is verified
  only by the consumer's unit tests.
- Source: `thesis/ingenieur/ch5_validation.tex:115–121`.
- Fix: add to §4.2 "No cancelled decode completed, so the ordering rule was never exercised by a
  trial; it is verified by unit tests."

**I-6 — "Loading the parse path" overstates the load** (l. 659; also l. 396–397; check 2)
- Problem: the loaded condition ran the language model only; no reflex trial ran during speech
  recognition. In addition, "added 12 ms" is the difference of the rounded p95s (18 − 6). The raw
  difference is 12.5 ms at p95, 6.9 ms at p50 and 5.8 ms on the mean.
- Source: `thesis/ingenieur/ch5_validation.tex:111–112`; `results/exp2.csv` (`a_system` rows).
- Fix: "loading the language model added 12 ms at p95 …", and add to Limitations: "no reflex trial
  overlapped speech recognition."

**I-7 — The recovery miss is missing from the abstract, highlights and conclusion** (l. 62–63, 75, 726–727; check 3)
- Problem: preemption is presented as a clean success ("All 78 … cancelled it"), yet recovery after
  preemption missed its budget by a factor of four (1,195 ms p95 against 300 ms).
- Source: `results/exp2_latency_budget.md` (Preemption recovery row, MISSES).
- Fix: add to the abstract "although recovery took 1,195 ms against 300 ms". The abstract is at
  250 words, so cut elsewhere: the validator/state-machine sentence (l. 61–62) or the membership
  sentence can be shortened by about 30 words.

**I-8 — The 540 ms is attributed to "the spotter" alone** (abstract l. 64, highlight l. 76, conclusion l. 730; check 3)
- Problem: the measurement cannot separate the spotter's decision delay, the 80 ms frame
  quantisation and the error of the energy-based offset anchor. The Results (l. 491) and the
  Discussion (l. 661–663) say this correctly; the abstract, highlight and conclusion do not.
- Source: `thesis/ingenieur/ch5_validation.tex:98–101`.
- Fix, highlight (78 chars): "Reflex latency misses its budget before the board; the board adds
  only 6-18 ms". Abstract: "almost entirely before the board's share (spotter delay, frame
  quantisation, offset estimate)".

**I-9 — The abstract does not state that the safe-failure target is missed** (l. 66–67; check 3)
- Problem: "failures did not shift towards safe outcomes" describes the trend but not the verdict
  (0.014–0.039 against ≥ 0.70 at every level).
- Fix: "… and the safe-failure rate stayed at 0.014–0.039 against a target of 0.70". This competes
  with I-7 for the word budget, so prioritise the verdicts.

**I-10 — One reflex false trigger on the golden recordings vs none on the same recordings is left unexplained** (l. 502–503 vs l. 522–523; check 3)
- Problem: §4.2 reports a `hover` fired during a landing command, and §4.3 reports zero false
  accepts on the golden-set recordings (0.304 h). A reader sees a contradiction.
- Source: `thesis/ingenieur/ch5_validation.tex:122–125` ("the records do not explain the
  difference"); `results/exp2_analysis.md:41`; `results/wake_training.md:26`.
- Fix: carry the thesis sentence over: the runtime replay fired once where the offline scoring of
  the same recordings did not, and the records do not explain why.

**I-11 — The keyword threshold is said to be selected "before the test split was scored"** (l. 362–363; check 2/3)
- Problem: the heads were fitted twice. The first round scored the test split and was rejected, so
  test was seen once before 0.999 was selected in round two.
- Source: `results/wake_training.md:40`; `thesis/ingenieur/ch4_implementation.tex:132–143`.
- Fix: "… which selected 0.999; the heads were fitted twice, and the test split had been scored once
  in the rejected first round."

**I-12 — Torkamani and Zarin's system routes between edge and cloud; it is not a local-only Jetson setup** (l. 143–144; check 6)
- Problem: the article lists it among systems that "run recognition offline beside a local model
  on … a Jetson board". The paper routes between a Jetson edge model and the cloud and prefers the
  cloud when it is available.
- Source: `thesis/ingenieur/ch3_state_of_the_art.tex:95,126` ("Edge (Jetson) or cloud, routed at run
  time").
- Fix: "\citet{torkamani2025} route between a model on a Jetson board and the cloud".

**I-13 — CommandSwarm is credited with "hold on rejection" and with not running on an SBC, neither of which the paper supports** (l. 704–706; check 6)
- Problem: CommandSwarm rejects commands. The thesis attributes fallback-to-hold to SkySim (and
  MIRA), not to CommandSwarm. CommandSwarm states no hardware ("unstated hardware" in the thesis
  table), so "neither runs on a single-board computer" is not supported for it.
- Source: `thesis/ingenieur/ch3_state_of_the_art.tex:400–408, 322–323`.
- Fix: "CommandSwarm shares our use of deterministic gates, and SkySim our hold on rejection; neither
  reports running on a single-board computer or separates its stop from the interpreting path."

**I-14 — `henry2026` may have a published version** (bib; check 6)
- Subagent 4 reports IEEE RO-MAN 2026, DOI 10.1109/RO-MAN72286.2026.11714651. The coordinator's
  search found the arXiv preprint (2606.24910) but did not confirm the venue, so verify the DOI before
  updating the entry.

### Minor

- **I-15** (check 3): "can preempt that model reliably" (conclusion, l. 726) should note that both
  keyword criteria are not demonstrated.
- **I-16** (check 3): l. 689–690 calls the endpointer splits "recoverable" at a CRR of 0.730.
  That figure still misses 0.80, and re-scoring is not a system change. Add "still below the 0.80
  target".
- **I-17** (check 3): the summary-table caption (l. 619) says "every requirement", but there is no
  safe-failure row; the miss is added only in the text (l. 650–651). Either add the row or say
  "every requirement of this paper".
- **I-18** (check 3): the abstract l. 65 "took 3,122 ms" should say "at the 95th percentile".
- **I-19** (check 2): l. 387 "replayed … in real time" holds for the latency experiment. The
  acoustic experiment replayed faster than real time (`thesis/ingenieur/ch5_validation.tex:255`;
  `results/table20_end_to_end.md`).
- **I-20** (check 2): the conclusion's first sentence (l. 725–727) and the gap paragraph (l. 173)
  should say "five simulated drones" and "from recorded audio".
- **I-21** (check 6): l. 155–157 says Simões et al. "select one". The paper concludes that the choice
  "depends on specific requirements" and has a third pipeline; the thesis table says only
  "Alternatives". Fix: "compare … as alternatives".
- **I-22** (check 6): l. 150–151 "None of these systems runs its speech and language chain on a
  single-board computer without a GPU". Three of the cited systems state no hardware, so write
  "None reports running …".
- **I-23** (check 6): `relays2s` is accepted to EMNLP 2026 Findings per its arXiv comment, with no
  DOI yet. Update it when the proceedings appear.
- **I-24** (check 7): interpretation inside Results: l. 481–482 ("because the deployed parser
  processes the whole prompt"), and l. 611 "needed" (the measurement is "intervened").
- **I-25** (check 7): measurements that first appear in Discussion: the synthetic spotter delay of
  131–188 ms (l. 665–666) and the recovery maximum (1,505 ms, l. 668), which Results never reports.
  Move them to §4.2.
- **I-26** (check 8): the abstract is at exactly 250 words. Any fix in I-7, I-9 or I-18 must be paid
  for by cuts.

---

## Cross-article

**X-1 — Two different word error rates for the same speaker** (should-fix; check 5)
- Master l. 241 and l. 670 give 23.1% (95% CI 20.3–25.9) as "the recogniser's word error rate on the
  golden speaker". Ingénieur Table 3, l. 543, gives 23.3% [20.2, 26.6] for clean audio.
- Both are sourced, but they measure different things. 23.1% is the workstation recogniser without
  the domain prompt (`results/table16_asr_speaker_sensitivity.md:24`); 23.3% is the full on-board parse
  path with the prompt and endpointer segmentation (`results/table20_end_to_end.md:15`).
- Fix: in the Master, say "23.1% (recogniser alone, without the domain prompt; 23.3% through the
  deployed pipeline [companion])", or quote the companion's 23.3% and cite it.

**X-2 — The 1,063 ms cited across articles** (should-fix): see M-8.

**X-3 — Parser failures on reference text restated without citation** (minor): Ingénieur l. 577–578,
"On the reference transcripts the parser made 13 failures, none of them safe", restates a
Master-side result without citing the companion.

**Verified consistent:**
- Shared figures agree in value and rounding: 0.935 EM, 3,122 ms over 226 segments, 622 ms (25%),
  751 ms uncached prefill, 328.8 ms cached prefill, the ≥ 0.70 safe-failure target, and 1 `unknown`
  in 2,640 in-domain decodes.
- The Ingénieur cites the Master for the model comparison and does not re-report it.
- No table or figure appears in both articles, and the contribution lists do not overlap. An 8-word
  verbatim-overlap scan found only boilerplate (section roadmap, one caption phrase, the board
  description). **No duplicate-publication risk found.**
- The companion entries are consistent: both are `@unpublished` with "Companion paper, in preparation",
  2026, and each title equals the other article's `\title`.
- Build logs: no undefined citations or references. Every cited key is in its bib, and no bib entry
  is uncited. Both bibs match `thesis/references.bib` field for field, apart from `urldate`→`note` and
  the new `companion` entry.

---

## Suggested order of work

1. Errors I-1, I-2 (fix the thesis too) and I-3.
2. Honesty items: M-1 to M-5, and I-4 to I-11. The Ingénieur abstract needs a word-budget pass for
   I-7, I-9 and I-18 together.
3. Citations and labels: M-6 to M-9, I-12 to I-14, X-1, X-2.
4. Minors, at the authors' discretion.

Subagent worksheets, with the full per-item tables (including every VERIFIED item and its
computation), are in the session notes folder: `master_numbers.md`, `ingenieur_numbers.md`,
`claims_consistency.md` and `citations.md`.

---

## Fixes applied (2026-10-13)

All findings above were applied to both articles, except the two below. Both PDFs rebuild with no
LaTeX errors and no unresolved references. The remaining BibTeX warnings are "empty pages" on
conference papers without page numbers (ICLR, NeurIPS, COLM, MLSys).

- **Not changed: I-14 (`henry2026`).** The RO-MAN 2026 venue could not be confirmed (only the arXiv
  preprint 2606.24910 was found), so the entry stays arXiv-only until the DOI is checked.
- **Not changed: I-23 (`relays2s`).** There is no proceedings version or DOI yet.
- **M-10 (`smollm2`):** confirmed on the COLM 2025 accepted-papers list (Spotlight). The entry now
  uses the COLM title and its OpenReview URL.
- **M-8 / X-2:** the Ingénieur latency text now reports the combined prefill-plus-decode p95
  (1,658 ms) and the shortest speech-recognition time (1,063 ms), so both Master figures cited to
  the companion paper can be found in it.
- **I-2, thesis:** the same sentence was also corrected in
  `thesis/ingenieur/ch3_architecture.tex:189–190`. `thesis/` is gitignored, so that edit is local only.
- **Abstract lengths after the fixes:** Master 242 words, Ingénieur 247 words. The Ingénieur
  abstract dropped its validator/state-machine sentence to make room for the recovery and
  safe-failure verdicts. All highlights are 69–81 characters.
