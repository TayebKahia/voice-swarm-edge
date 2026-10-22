# Step 1 audit: verification of the Master and Ingénieur journal articles

Date: 2026-10-22. Scope: verification against the source material, not peer review. Each
finding below is either wrong, unsupported, inconsistent, or missing relative to the compiled
theses, `results/` or the cited works. No file was changed during the audit.

- Articles: `articles/master/main.tex` (M), `articles/ingenieur/main.tex` (I). Line numbers are
  those of the files at commit `3c1e0f7`.
- Theses (authoritative): `main_master.pdf` (79 pp) and `main_ingenieur.pdf` (90 pp).
  **Page numbers are physical PDF pages** (the printed page number is one less in the body).
  Written below as `M.pdf p.N` and `I.pdf p.N`.
- Raw data: `results/`, plus `data/wake/wake_manifest.json` where noted.

## How the audit was done

- **Phase A, parallel checks.** Seven read-only subagents ran in parallel (Sonnet, high effort):
  numbers in M, numbers in I, citations, fidelity and scope, figures and tables, build and template,
  language and consistency.
  - Figures were regenerated in a scratch copy of the repository (`python eval/plots.py --article`).
  - Both articles were compiled in a scratch directory.
  - Every DOI was resolved through Crossref, and arXiv, GitHub and publisher pages were fetched.
- **Phase B, re-verification.** I re-checked each subagent finding against the cited article line
  and the cited thesis page or data file before it entered this report.
  - Duplicates found by several subagents are merged; the source IDs are given in brackets.
  - Rejected and unverifiable findings are in the Appendix.
- **Phase C, comparison.** `REVIEW_verification.md` was opened only after Phases A and B.

## 1. Summary

**No numeric errors.** Every number in both articles traces to the thesis PDFs or to
`results/`:
- about 150 items in M and about 120 in I;
- every table cell;
- every figure: all ten committed figure PDFs are byte-identical to the regenerated ones.

**Both articles build cleanly:** 0 LaTeX errors, 0 undefined references or citations, 0
missing bib keys, 0 unused bib entries. The repository named in Data availability is public, and
its `v1.0.0` release carries `qwen2.5-0.5b-instruct-Q4_K_M.gguf` (GitHub API, 2026-10-22).

The problems lie in framing, dropped caveats, provenance, definitions and presentation.

| Article | error | should-fix | minor |
|---|---|---|---|
| Master (M-) | 0 | 13 | 21 |
| Ingénieur (I-) | 1 | 14 | 25 |
| Cross-article (X-) | 0 | 2 | 4 |

**The three most important problems**

1. **I-01: the Ingénieur article claims contributions that its thesis explicitly disclaims.**
   - The article lists four contributions: the dual-path runtime, the membership rule, the
     validation chain, and the evaluation.
   - The thesis claims one contribution, the membership rule with its own reflex budget. It states
     that running a fast path beside the model "is not itself claimed".
   - The thesis also says that reporting the engineering as a research contribution "would
     overstate the claim" (I.pdf p.21, p.85).
2. **M-01 and M-02: the Master article's headline wording is stronger than its thesis.**
   - **M-01:** "Model family alone" changes EM by 6.5 points (abstract, highlight, contribution).
     The thesis calls this contrast exploratory and confounded with prompt rendering and training
     session (M.pdf p.68).
   - **M-02:** The conclusion's "best accuracy–efficiency trade-off" contradicts the thesis
     verdict. The thesis finds that the trade-off "does not reach the deployment decision" and that
     selection is on accuracy alone (M.pdf p.65).
3. **X-01: the anchor of the 3,122 ms end-to-end figure is never defined.**
   - Both articles call it "from the end of speech". It is measured from the endpointer's
     declaration (T0), which comes after the 480 ms endpointing wait.
   - The Ingénieur article lists that wait inside a budget whose allowances after T0 already sum to
     2,600 ms > 2,500 ms, with no explanation (I.pdf p.51–52).

---

## 2. Findings

Format: **ID — title** (severity), location, evidence, problem, proposed replacement. "Replace"
gives the exact old → new text; ellipses (…) mark unchanged surrounding text.

### 2.1 Master article (`articles/master/main.tex`)

#### Should-fix

**M-01 — "Model family alone" is unqualified in the abstract, highlight and contribution** [NM-01, FS-M1]
- Location: l.65–66, l.80, l.130–131.
- Evidence: M.pdf p.68 (§6.3.1 "should be quoted with them"; the Qwen/Danube pair also differs in
  system-prompt rendering and training session); p.68–69 (fixed budget, no model converged); p.57
  (exploratory test). The body (l.262–269, 416, 623–627, 683–685) keeps these caveats; the front
  matter drops them.
- Replace:
  - l.65–66: "At matched size, family alone separates Qwen2.5-0.5B from H2O-Danube3-500M by 6.5
    percentage points ($p = 0.0023$)." → "At matched size, Qwen2.5-0.5B exceeds H2O-Danube3-500M by
    6.5 percentage points (exploratory, $p = 0.0023$), a contrast that also differs in prompt
    rendering and training session."
  - l.80: "At matched size, model family alone changes exact match by 6.5 points" → "At matched
    size, two model families differ by 6.5 points in exact match" (68 characters).
  - l.130–131: "showing that, at about 0.5~B parameters, model family alone changes exact match by 6.5
    percentage points." → "showing that, at about 0.5~B parameters and under one training budget,
    two model families differ by 6.5 percentage points in exact match."

**M-02 — The conclusion's "best accuracy–efficiency trade-off" is not the thesis verdict** [FS-M2]
- Location: l.695–697.
- Evidence: M.pdf p.65 ("RQ1's trade-off exists on the figure, but under a rule of this form it
  does not reach the deployment decision"; latency and memory are constraints and accuracy is the
  only ranking criterion). I.pdf p.85 ("one of two non-dominated configurations … the best measured
  on its selection criteria against re-baselined constraints").
- Replace l.695–697: "…Qwen2.5-0.5B at \QF{} offers the best accuracy--efficiency trade-off for
  grammar-constrained command parsing: 0.935 golden-set EM at 27.93~tokens/s on a cooled board." →
  "…Qwen2.5-0.5B at \QF{} is the configuration the selection rule retains for grammar-constrained
  command parsing: 0.935 golden-set EM at 27.93~tokens/s on a cooled board. It and SmolLM2-360M are
  the two non-dominated points, and the rule selects between them on accuracy alone."

**M-03 — The chat-template parity check is overstated** [NM-02, FS-M5]
- Location: l.288.
- Evidence: M.pdf p.45. The llama.cpp half of the check confirmed the three models at Q4_K_M only:
  "The Q8_0 artefacts were not run through it, but their tokeniser and chat-template metadata are
  identical". Llama-3.2's date re-alignment "accepts one real difference by design".
- Replace l.288: "The three deployment candidates pass on all ten." → "The three deployment
  candidates pass on all ten at \QF{}; the \QE{} artefacts were not run through the check, but their
  tokeniser and chat-template metadata are identical to their \QF{} counterparts'. Llama-3.2's
  template writes the current date into the system turn, so its served prompt differs from the
  training prompt by that date."

**M-04 — 60 timing repetitions instead of the 200 planned, with narrow margins** [FS-M6]
- Location: l.302–305; Limitations l.685–686.
- Evidence: M.pdf p.46 ("The protocol as designed called for 200 repetitions per configuration.
  The multi-model benchmark timed each Q4_K_M configuration over 60"). The margins are narrow:
  decode p95 1,033 ms against 1,100 ms (l.524), and 278 ms spare (l.528).
- Replace:
  - l.303: "then timed over 60 scored trials" → "then timed over 60 scored trials (the protocol had
    planned 200)".
  - Limitations l.685–686: "Timing comes from one board and one runtime build, with a cached
    prefill." → "Timing comes from one board and one runtime build, with a cached prefill, and from
    60 repetitions per configuration, so a margin as narrow as the selected model's decode p95
    (1,033~ms against 1,100~ms) could move with a few more slow trials."

**M-05 — The golden speaker's profile and the speaker-sensitivity result are omitted** [FS-M4]
- Location: l.241–244, l.679.
- Evidence: M.pdf p.53 (Table 5.1: "Author (L2 English, domain commands)" 23.1 [20.3, 25.9]; across
  15 Common Voice accent buckets WER ranges from 6.2% [0.0, 16.3] to 49.5% [35.0, 63.6]); p.70
  (transfer to other speakers not established).
- Replace l.679: "The golden set is one speaker in one session," → "The golden set is one non-native
  English speaker in one session; across 15 accent groups of Common Voice speech the recogniser's
  word error rate ranges from 6.2\% to 49.5\%, so transfer to other speakers is not established. Its
  200 items are paraphrases"
  - This also restructures the sentence: the original continues "and its 200 items are paraphrases".

**M-06 — The Limitations section understates the loss from audio** [FS-M3]
- Location: l.677–679.
- Evidence: M.pdf p.70 (§6.3.2); I.pdf p.72 (Table 6.2: CRR 0.690 clean and 0.590 at 10 dB,
  against EM 0.935 on text).
- Replace l.678–679: "…overstate what the pipeline achieves from audio; the word error rate on the
  golden speaker is 23.3\% through the deployed pipeline~\citep{companion}." → "…overstate what the
  pipeline achieves from audio: through the deployed pipeline the word error rate on the golden
  speaker is 23.3\%, and the command recognition rate is 0.690 in clean audio and 0.590 at 10~dB of
  rotor noise, against 0.935 EM on the reference transcripts~\citep{companion}."

**M-07 — Memory figures are listed in a different order from the sentence before them** [NM-06, FS-M9, LC-18]
- Location: l.530–531.
- Evidence: The two preceding sentences list Qwen, SmolLM2, Llama. Read in that order, the list
  gives Qwen 0.55 GiB, which is wrong. M.pdf p.54: Qwen 0.68, SmolLM2 0.55, Llama 1.63.
- Replace: "Peak resident memory of the language-model process is 0.55, 0.68 and 1.63~GiB for the
  three models." → "Peak resident memory of the language-model process is 0.68~GiB for Qwen2.5-0.5B,
  0.55~GiB for SmolLM2-360M and 1.63~GiB for Llama-3.2-1B."

**M-08 — "More than 40% of the span" is not in the thesis and contradicts the next sentence** [NM-09, FS-M10]
- Location: l.620–621.
- Evidence: M.pdf p.64–65. The thesis says "more than a third" of the 16.0 pp fp16 Qwen–SmolLM2
  gap. Article l.623–624 itself says "a family effect cannot be expressed as a share of a size
  effect".
- Replace: "At matched size, the family difference (6.5\pp{}) is more than 40\% of the span across
  the whole 3.4-fold size range (15.0\pp{})." → "At matched size, the family difference (6.5\pp{}) is
  of the same order as the 15.0\pp{} span across the whole 3.4-fold size range, whose endpoints also
  differ in family."

**M-09 — The sources of the training audio and noise are neither named nor cited** [CI-09]
- Location: l.229–231.
- Evidence: M.pdf p.41: the audio is Piper TTS, and the noise comes from DREGON plus the ESC-50
  helicopter and engine classes, all cited in the thesis.
- Replace: "…replaced by the recogniser's output on synthesised speech, mixed with noise at 20, 10 or
  5~dB signal-to-noise ratio," → "…replaced by the recogniser's output on speech synthesised with
  Piper~\citep{piper}, mixed with rotor and engine noise from DREGON and ESC-50~\citep{dregon,esc50}
  at 20, 10 or 5~dB signal-to-noise ratio,"
- Bib: add `piper` and `dregon` (copy from `articles/ingenieur/references.bib`) and `esc50` (copy
  from `thesis/references.bib`).

**M-10 — Abbreviations are used without expansion** [LC-01–04]
- Location: l.256 (LoRA), l.128 (GGUF), l.214 (GBNF), l.242 (CI), l.174 (NLP).
- Evidence: grep shows no expansion of any of the five in the body. The abstract has "low-rank
  adaptation" but not "LoRA".
- Replace:
  - l.256: "the same LoRA recipe" → "the same low-rank adaptation (LoRA) recipe".
  - l.128: "at two GGUF quantisation levels" → "at two quantisation levels of the GGUF file format of
    \texttt{llama.cpp}".
  - l.214: "in the GBNF formalism of" → "in the GBNF (GGML Backus--Naur form) formalism of".
  - l.242: "(95\% CI 20.3--25.9)" → "(95\% confidence interval, CI, 20.3--25.9)".
  - l.174: "structured NLP tasks" → "structured natural-language-processing tasks".

**M-11 — "Abstention" is a headline term but is never defined** [LC-05]
- Location: Metrics l.318–321; used at l.128, 138, 552, 664.
- Evidence: Metrics defines only the safe-failure and false-command rates.
- Replace: insert after l.321 ("…anything other than \texttt{unknown}."): "We call an output of
  \texttt{unknown} or \texttt{hover} an abstention; the two rates measure abstention on wrong
  outputs and on out-of-domain inputs."

**M-12 — The throughput definition is ungrammatical** [LC-06]
- Location: l.320–321.
- Replace: "Throughput is decode tokens summed over decode time summed." → "Throughput is the sum of
  decode tokens divided by the sum of decode time."

**M-13 — Labels overlap data in the thermal figure** [FT-12]
- Location: `figures/fig_thermal.pdf` (l.538).
- Evidence: Rendered figure. The "decode allowance 1,100 ms" text is drawn across both Llama-3.2-1B
  bars in the top panel. The "18.1" label (Qwen, uncooled) is struck through by the 20 tokens/s
  line. The data are correct (`results/thermal_headroom.md`).
- Fix: in `eval/plots.py` (`article_thermal` / `_paired_panels`):
  - place the target labels where they clear the bars (two-line label at the right edge);
  - give value labels that collide with a target line a white background or an offset;
  - regenerate with `--article`.

#### Minor

**M-14 — The training libraries are named without citation** [CI-08]
- Location: l.261.
- Evidence: the thesis cites `transformers`, `peft` and `pytorch` (thesis/references.bib).
- Replace: "with \texttt{peft} 0.19.1, \texttt{transformers} 5.0.0 and PyTorch 2.10.0" → "with
  \texttt{peft} 0.19.1~\citep{peft}, \texttt{transformers} 5.0.0~\citep{transformers} and PyTorch
  2.10.0~\citep{pytorch}". Copy the three entries from `thesis/references.bib`.

**M-15 — "Perplexity" is attributed to `kurtic2025`** [CI-05]
- Location: l.164–166.
- Evidence: the full text of arXiv 2411.02355 has no perplexity evaluation; `kurt2026` does.
- Replace: "Their cost is evaluated as perplexity and accuracy on general-purpose suites:" → "Their
  cost is evaluated as accuracy on general-purpose suites, and for the \texttt{llama.cpp} formats
  also as perplexity:".

**M-16 — The bootstrap EM intervals post-date the thesis and their provenance is not stated** [FS-M8]
- Location: l.334–335.
- Evidence: M.pdf p.54 (Table 5.2 has no intervals); the values are `results/em_ci.csv` and
  `eval/em_ci.py`, which match exactly.
- Fix: none required in the text, because Data availability covers the scripts. Mention in the
  cover letter or to the supervisor that the intervals are a post-thesis addition.

**M-17 — Contribution 4 conflates the rule's failure with the abstention targets** [FS-M13]
- Location: l.137–138.
- Replace: "and a selection rule that applies the deployment constraints first and reports its own
  failure, including two abstention targets that no configuration meets." → "and a selection rule
  that applies the deployment constraints first and reports its own failure, beside two abstention
  targets that no configuration meets."

**M-18 — The quantisation scope is not stated in the Limitations** [FS-M14]
- Location: l.688–689.
- Evidence: M.pdf p.71.
- Replace: "Finally, the schema is small:" → "Only two GGUF quantisation levels were examined; lower bit
  widths and other quantisation methods were not measured. Finally, the schema is small:"

**M-19 — Two golden-set facts are omitted** [FS-M15, partly]
- Location: l.237–238, Limitations l.681–683.
- Evidence: M.pdf p.41. One of the 12 patterns (`set_param` with altitude and spacing) never
  occurs in training. The two further planned sessions (noisy room, later day) were not recorded.
- Replace l.238: "items each." → "items each; one of these patterns, \texttt{set\_param} with
  altitude and spacing, never occurs in training."

**M-20 — "Bounding the grammar mattered as much as having one" is a new comparative claim** [FS-M16]
- Location: l.638.
- Evidence: M.pdf p.12, p.38. The only evidence is a five-command zero-shot run under an early
  grammar.
- Replace: "Bounding the grammar mattered as much as having one." → "Bounding the grammar also
  mattered."

**M-21 — A heading conflicts with the article's own synthetic and out-of-domain results** [FS-M10]
- Location: l.617; l.392–393.
- Evidence: l.393 says that on those sets "the ordering follows parameter count".
- Replace: "\paragraph{Parameter count does not order these models.}" → "\paragraph{On the golden
  set, parameter count does not order these models.}"

**M-22 — LLM2Swarm's small-model test sits beside "rely on … multi-billion parameter models"** [FS-M17]
- Location: l.156–157.
- Replace: "These systems rely on remote or multi-billion parameter models, and none compares" →
  "These systems rely on remote or multi-billion parameter models, or test a small model for
  generation speed only, and none compares".

**M-23 — The preliminary test anecdote drops its scope** [NM-07]
- Location: l.108–110.
- Evidence: M.pdf p.12: five commands, 48-token cap, two looped; the truncated outputs "resolved
  to a hover, so the landing was not executed".
- Replace: "In a preliminary zero-shot test on the target board, Qwen2.5-0.5B under an early grammar
  repeated drone identifiers for the command ``swarm land'' until the token cap truncated the output;
  the truncated output failed to parse, and the landing was lost." → "In a preliminary zero-shot test
  of five commands on the target board, Qwen2.5-0.5B under an early grammar repeated drone
  identifiers on two of them, including ``swarm land'', until the token cap truncated the output; the
  truncated output failed to parse and resolved to a hover, so the landing was not executed."

**M-24 — Half-way values are rounded in opposite directions** [NM-04, FT-18]
- Location: Table 1 l.379–380, 383; l.418; l.504.
- Evidence: `results/em_ci.csv`. Values are rounded as follows: 0.8125 → 0.812, 0.8875 → 0.887, but
  0.9625 → 0.963. The thesis tables print the same values.
- Fix (optional): leave as is to match the thesis, or print these three at four decimals (0.8125,
  0.8875, 0.9625).

**M-25 — The latency-stack figure is wider than the column, so its text prints at about 5 pt** [FT-13]
- Location: `figures/fig_latency_stack.pdf` (l.512).
- Evidence: the PDF is 280 pt wide against a 238 pt column and is scaled to 0.85.
- Fix: in `eval/plots.py`, shorten the x-label (e.g. "Language-model time per command (ms)") so the
  saved figure is at most 238 pt wide; order the legend prefill, decode, total.

**M-26 — The bold cell in Table 1 is unexplained** [FT-14]
- Location: l.369–370.
- Replace: "Schema validity is 1.000 in every cell and is omitted." → "Schema validity is 1.000 in every
  cell and is omitted. Bold: the selected configuration."

**M-27 — The sign convention of Δ is not stated** [FT-15]
- Location: l.428–429.
- Replace: "Precision, runtime, grammar and decoding settings change together." → "$\Delta$: deployed
  minus fp16. Precision, runtime, grammar and decoding settings change together."

**M-28 — The Pareto figure's memory labels are undefined, and its allowance line is named differently** [FT-16]
- Location: l.586–591; `figures/fig_pareto.pdf`.
- Replace: in the caption, after "filled markers are non-dominated." insert "Point labels give the p95
  and the peak resident memory of the language-model process." In `eval/plots.py`, rename the
  in-figure "stage allowance" label to "language-model allowance".

**M-29 — The abstract's last sentence is hard to parse** [LC-13]
- Location: l.71–74.
- Replace: "A selection rule that applies the deployment constraints first selects Qwen2.5-0.5B at
  Q4\_K\_M only after re-baselining the latency budget: its end-to-end latency … or the
  safe-failure target." → "Applying the deployment constraints first, a selection rule retains
  Qwen2.5-0.5B at Q4\_K\_M only after re-baselining the latency budget, because its end-to-end latency
  (3,122~ms at the 95th percentile) exceeds the 2,500~ms budget. No configuration meets the
  false-command target on out-of-domain input ($\leq$~0.05; measured 0.19--0.39) or the safe-failure
  target."

**M-30 — Percentage points are named three ways** [LC-07]
- Location: l.66, l.68, l.80.
- Replace l.68: "$\pm$1.5 points" → "$\pm$1.5 percentage points". l.80 is covered by M-01.

**M-31 — Methods and Results switch tense** [LC-08, LC-09]
- Location: l.302–306; Results l.389–396, l.521–531 against l.545–550.
- Replace l.302–303: "is warmed up for ten minutes and then timed" → "was warmed up for ten minutes
  and then timed". Further tense harmonisation is at the authors' discretion.

**M-32 — Parameter counts are formatted inconsistently** [LC-15]
- Location: l.79 ("0.5B", "1.2B"), l.127 ("0.36--1.24~B"), l.617 ("0.49~B", "1.24~B").
- Fix: keep the highlight (character limit) and use "~B" with a thin space in the body throughout.

**M-33 — Small language items** [LC-10, 11, 12, 14, 16, 17]
- l.99 "onboard computer" → "on-board computer".
- l.571 "with a hold" → "with a \texttt{hover}".
- l.624 "And the three-epoch budget" → "Moreover, the three-epoch budget".
- l.645 "cheap insurance" → "a low-cost safeguard".
- l.377 header "tok/s" → "tokens/s".
- l.306 "The server's prompt cache" → "The \texttt{llama.cpp} server's prompt cache".
- l.279 "Their difference" → "The difference between the two surfaces".

**M-34 — Bibliography completeness** [CI-07, BT-08]
- `geng2023`: add `doi = {10.18653/v1/2023.emnlp-main.674}` (Crossref).
- `qin2021`: add `doi = {10.24963/ijcai.2021/622}` (Crossref).
- Seven BibTeX "empty pages" warnings remain on conference papers without pages (ICLR, NeurIPS,
  COLM, MLSys). No action needed.

### 2.2 Ingénieur article (`articles/ingenieur/main.tex`)

#### Error

**I-01 — The contributions list claims what the thesis disclaims** [FS-I1]
- Location: l.110–125.
- Evidence: I.pdf p.21 (§1.5): "Of the four contributions claimed across the two documents, one
  belongs to this document". The engineering ("an offline voice pipeline …, a validated command bus,
  a flight state machine and a swarm controller") is described as "None of them is claimed as novel …
  reporting it as a research contribution would overstate the claim". I.pdf p.85: "Running a fast
  path beside the language model is not itself claimed". I.pdf p.41–42 concedes preemption to MIRA.
  The article lists four contributions, including the runtime, the validation chain (whose first
  layer, the grammar, is the Master's) and the evaluation. It also calls deterministic gating prior
  art at l.149–150.
- Replace l.110–125, from "The contributions are:" through the end of the itemize, with:

  ```
  The contribution we claim is narrow:
  \begin{itemize}
  \item A membership rule for a path that bypasses interpretation: a command qualifies only if its
    delay changes the physical outcome and its unprompted execution can only make the swarm do less.
    For the ten intents of the schema the rule admits exactly two, a hold and an abort, and it states
    the condition under which a third class could be admitted.
  \item A latency budget of that path's own, specified per stage and measured on the board while
    the language model decodes on the same processor.
  \end{itemize}
  To measure them we built a dual-path runtime in which a keyword-spotter \emph{reflex path}
  preempts language-model inference on the \emph{parse path}, with an ordering rule that prevents a
  stale command from overriding a stop, and a three-layer validation chain. These are engineering,
  not claimed as novel: preemption of a slower path by a fast gate is prior art~\citep{mira}, and
  the decoding grammar is the companion paper's. We evaluate both paths against the budget, idle and
  under load, together with keyword-spotter error rates, recognition under rotor noise including the
  direction of the errors, and formation control of five simulated drones, and report every missed
  target.
  ```

  - The highlights already lead with the rule (l.74), so they need no change.

#### Should-fix

**I-02 — An unfavourable comparison with MIRA is omitted** [FS-I2]
- Location: l.159–162; Discussion l.712–719.
- Evidence: I.pdf p.36 (MIRA's gate: 466 ms median from speech onset); I.pdf p.85 ("Measured from
  the onset of the keyword, the reflex takes 965 ms at the median … What the reflex adds over that
  gate is therefore not speed but a trigger restricted to two commands by a derived rule, and a
  budget").
- Replace: append to the end of the "Comparison by design" paragraph (after l.719): "MIRA's gate
  aborts at a median of 466~ms from speech onset~\citep{mira}; our reflex path took 965~ms at the
  median from keyword onset. What it adds over that gate is therefore not speed but a trigger
  restricted to two commands by a derived rule, and a budget against which its latency is located."

**I-03 — The recogniser's domain prompt is not mentioned** [NI-2]
- Location: l.238–239.
- Evidence: I.pdf p.54: "The recogniser's invocation, including an initial prompt that lists the
  command vocabulary". All WER, CRR and safe-failure figures are measured with it
  (`results/exp3_pi_analysis.md`). The Master article l.242 contrasts the WER "without its domain
  prompt", so the Ingénieur side should name it.
- Replace l.239: "\texttt{tiny.en} model transcribes each one~\citep{whisper,whispercpp}," →
  "\texttt{tiny.en} model transcribes each one with an initial prompt that lists the command
  vocabulary~\citep{whisper,whispercpp},".

**I-04 — The failure-cause counts do not reconcile in the text** [NI-1]
- Location: l.574–576, l.587–588.
- Evidence: I.pdf p.72 ("23 failed, and one of them is counted under a cause earlier in the
  order"). The text says 23 split and all failed, but attributes 22. It also says 12 parser errors
  at l.576 and 13 failures on reference text at l.588.
- Replace l.574–576: "endpointer splits accounted for 22 (23 of the 200 utterances, 11.5\%, were
  split at a pause, and all 23 failed), recognition errors … and parser errors already present on
  the reference text for 12." → "parser errors already present on the reference text accounted for
  12, endpointer splits for 22 (23 of the 200 utterances, 11.5\%, were split at a pause and all 23
  failed; one is counted as a parser error), recognition errors that left the intent right but a slot
  wrong for 23, and recognition errors that changed the intent for 5."
- The 13 reference-text failures against 12 parser-error causes: one reference failure passes from
  audio (`results/exp3_pi_preds`, item 0113). This is not in the thesis text, so it is left
  unstated.

**I-05 — A known state-machine defect is omitted** [FS-I5]
- Location: Limitations l.721–733.
- Evidence: I.pdf p.61 ("Legality of the original intent … legality is therefore judged on the
  command's original intent, not on the hold it became"); I.pdf p.83 (future work: correct it).
- Replace: insert before "Finally, the 20 recorded takes" (l.732): "The state machine judges legality
  on a command's intent before the validator substitutes a hold, so an unvalidated command could
  dispatch a hold in a state where a hold is illegal; on the deployed route validation comes first,
  so the case does not arise there."

**I-06 — The abstract does not say that formations were not commanded by voice** [FS-I7]
- Location: l.67–69.
- Evidence: I.pdf p.61–62 (link to the controller not built); article l.429–430 and l.726–727.
- Replace: "Five simulated drones converged in all 150 formation trials without collision, which rests
  on a separation clamp." → "Commanded directly rather than by voice, five drones in a kinematic
  simulator converged in all 150 formation trials without collision, a result that rests on a
  separation clamp."
  - This adds about 8 words; with I-20 the abstract reaches about 257 words. Trim elsewhere if the
    target journal caps it at 250 (e.g. drop "with no class confusion" at l.61; see I-22).

**I-07 — Synthetic false-reject rates and the stress stream lack their thesis caveats** [FS-I4]
- Location: l.532–534.
- Evidence: I.pdf p.70 ("Every positive of both splits is rendered by the same three synthetic
  voices the heads were trained on, so these rates measure the heads on voices they have heard");
  I.pdf p.71 (stress stream "not an operating rate, and no criterion is judged on it").
- Replace:
  - l.533: "34 false accepts in 0.100~h, 31 of them on the hold head." → "34 false accepts in
    0.100~h, 31 of them on the hold head; it is a stress condition, not an operating rate, and no
    criterion is judged on it."
  - l.534: "On the synthetic test takes, the false-reject rate was 0.027 for hold and 0.007 for
    abort;" → "On the synthetic test takes, rendered by the three voices used in training, the
    false-reject rate was 0.027 for hold and 0.007 for abort;".

**I-08 — The keyword-curve figure suggests that the false-reject target is met** [FT-04]
- Location: caption l.522–525; `figures/fig_keyword_curve.pdf`.
- Evidence: I.pdf p.70: "The false-reject criterion is judged at the same threshold on the
  author's recorded takes, because the curve's positives are synthetic". The figure draws the 0.10
  target across synthetic rates of 0.027 and 0.007, while the verdict is "not demonstrated".
- Replace: append to the caption: "The false-reject criterion is judged on the author's recorded
  takes (Section~\ref{sec:res-kws}), not on these synthetic takes."

**I-09 — The summary table claims p99, which the article never shows** [FT-01]
- Location: l.643.
- Evidence: Table 2 (l.451–460) has p50 and p95 only. The p99 values exist in I.pdf p.68.
- Replace: "All five stages, p50/p95/p99" → "All five stages, p50 and p95".

**I-10 — The trajectory figure has no generating script, although Data availability claims scripts for every figure** [FT-02]
- Location: l.597; l.765–768.
- Evidence: `figures/fig_trajectories.pdf` is byte-identical to
  `results/figure_formation_trajectories.pdf`. No code in the repository produces it, and
  `eval/plots.py --article` does not write it. Its content is consistent with trial 1 (seed 42) in
  `results/exp4.csv`.
- Fix: either commit the plotting function (e.g. `article_trajectories()` in `eval/plots.py`), or
  weaken l.767–768: "the scripts that produce every table and figure" → "the scripts that produce
  every table and the plotted figures".
- Also make the caption self-contained: add "Trial 1 (seed 42) of each formation; colours
  identify vehicles."

**I-11 — Abbreviations are used without expansion** [LC-20, LC-21]
- Location: l.195/238 (VAD), l.241 (GBNF), l.199/252 (UDP), l.376 (PID), l.422 (ANOVA), l.453/503
  (p50, p99).
- Replace:
  - l.237–238: "a voice-activity model (Silero VAD;~\citealp{silero})" → "a voice-activity detection
    (VAD) model, Silero VAD~\citep{silero},".
  - l.241: "bounded GBNF grammar" → "bounded grammar in the GBNF (GGML Backus--Naur form) formalism".
  - l.252: "over UDP" → "over the User Datagram Protocol (UDP)".
  - l.376: "a PID law" → "a proportional--integral--derivative (PID) law".
  - l.422: "a one-way ANOVA" → "a one-way analysis of variance (ANOVA)".
  - l.343: "at the 95th percentile (p95)" → "at the 95th percentile (p95; p50 and p99 denote the
    median and the 99th percentile)".

**I-12 — The 850 ms onset target appears only in Results** [NI-6, LC-23]
- Location: l.346–348 against l.503–504.
- Evidence: I.pdf p.51 (Table 4.3, "Reflex path from keyword onset 850 ms") and p.52 ("The onset
  figure is reported beside it as the time the operator perceives").
- Replace l.346: "The reflex path is budgeted at 150~ms from the keyword offset," → "The reflex path
  is budgeted at 150~ms from the keyword offset (and 850~ms from its onset, the delay the operator
  perceives),".

**I-13 — An ill-formed sentence in the Discussion** [LC-25]
- Location: l.689–691.
- Replace: "an optimised acoustic model or a streaming encoder could reduce it, at a possible cost in
  accuracy that would have to be selected on a held-out recording set rather than on the golden set,
  which is the test set." → "an optimised acoustic model or a streaming encoder could reduce it, at a
  possible cost in accuracy; such a change would have to be selected on a held-out recording set
  rather than on the golden set, which is the test set."

**I-14 — A dangling relative clause in the setup** [LC-24]
- Location: l.397–399.
- Replace: "…described in the companion paper~\citep{companion}, which the endpointer split into 226
  segments; each segment is one latency sample." → "…described in the companion
  paper~\citep{companion}; the endpointer split them into 226 segments, and each segment is one
  latency sample."

**I-15 — `henry2026` is now published** [CI-01; earlier review I-14]
- Location: `references.bib` l.191–201.
- Evidence: Crossref `10.1109/RO-MAN72286.2026.11714651` gives the same title and the same five
  authors, in "2026 IEEE 35th International Conference on Robot and Human Interactive Communication
  (RO-MAN)", pp. 2455–2460. Re-checked by me on 2026-10-22.
- Replace the entry with:

  ```
  @inproceedings{henry2026,
    title     = {End-to-End Voice Intent Recognition for Spontaneous Human-Drone
                 Interaction with Naive Users},
    author    = {Henry, Allan and Rossato, Solange and Graff, Christian and
                 Huet, Sylvain and Gomez-Balderas, Jose-Ernesto},
    booktitle = {2026 IEEE 35th International Conference on Robot and Human
                 Interactive Communication (RO-MAN)},
    pages     = {2455--2460},
    year      = {2026},
    publisher = {IEEE},
    doi       = {10.1109/RO-MAN72286.2026.11714651}
  }
  ```

#### Minor

**I-16 — `relays2s` was accepted at a venue** [CI-02; earlier review I-23]
- Evidence: the arXiv 2603.23346 comment reads "EMNLP 2026 Findings"; no DOI yet.
- Fix: add `note = {Accepted to Findings of EMNLP 2026}`.

**I-17 — The acoustic experiment's thermal state is not given** [NI-3]
- Location: l.413–427.
- Evidence: I.pdf p.73: 54.3–80.1 °C; the soft-temperature-limit flag was set during the 5 dB
  condition.
- Replace: append to the paragraph (after l.427): "The processor stayed at 54.3--80.1~$^\circ$C; its
  soft temperature-limit flag was set during the 5~dB condition, which affects timing, not
  accuracy."

**I-18 — Temperature was sampled at the start of each trial** [NI-4]
- Location: l.391.
- Evidence: `results/exp2_analysis.md` ("at trial start").
- Replace: "the processor stayed at 50.5--77.9~$^\circ$C" → "the processor, sampled at the start of
  each trial, stayed at 50.5--77.9~$^\circ$C".

**I-19 — "Takes" are counted as trials** [NI-8]
- Location: l.508–509; l.400–401 defines 20 takes per class, each replayed twice.
- Replace: "all 40 abort takes fired the abort class and 38 of 40 hold takes fired the hold class, the
  two misses being one take on both replays;" → "all 40 abort trials fired the abort class and 38 of
  40 hold trials fired the hold class, the two misses being one take on both replays;".

**I-20 — The abstract's recovery figure lacks its percentile** [NI-11]
- Location: l.61–62.
- Replace: "but recovery took 1,195~ms against 300~ms." → "but recovery took 1,195~ms at the 95th
  percentile against 300~ms."

**I-21 — No keyword-spotter verdict appears in the abstract** [NI-12]
- Location: l.53–70.
- Evidence: both keyword criteria are "not demonstrated" (Table 5).
- Fix: optional, given the word count. Add "Neither keyword-spotter error criterion was
  demonstrated." if space allows.

**I-22 — "No class confusion" holds for the recorded takes only** [FS-I3]
- Location: l.61, l.669–670, l.639.
- Evidence: I.pdf p.70: on the synthetic test split "One hold take fired the abort class".
- Replace l.669–670: "the two classes were never confused" → "the two classes were never confused on
  the recorded takes".

**I-23 — The "6–18 ms" measurement span ends at the bus** [FS-I8]
- Location: l.742.
- Replace: "the board added only 6--18~ms to the stop" → "the board added only 6--18~ms before the stop
  was published on the bus".

**I-24 — The ANOVA is presented first, without the paired-data caveat** [FS-I10]
- Location: l.582–586.
- Evidence: I.pdf p.72–73. The same 200 utterances appear in every condition. Two boundary pairs
  sit at p = 0.0051 and 0.0053 against α = 0.0050.
- Replace: after "rejected equality … 20 and 15~dB." insert "The five conditions reuse the same 200
  utterances, so the paired tests below are the primary analysis."

**I-25 — The frame-loop maximum during recognition is omitted** [FS-I11]
- Location: l.502–503.
- Evidence: I.pdf p.68: "In the golden-set phase, where it did run, no frame exceeded the period
  either, the longest taking 74.2 ms".
- Replace: after "and 45.5~ms at most." insert "While speech recognition ran, the longest frame took
  74.2~ms."

**I-26 — "Guarantee" contradicts the thesis's explicit disclaimer** [FS-I13]
- Location: l.708.
- Evidence: I.pdf p.18 ("not a claim that no collision could occur").
- Replace: "the same guarantee would have to act" → "the same mechanism would have to act".

**I-27 — The companion paper's selection is described too broadly** [FS-I14]
- Location: l.243–244.
- Evidence: M.pdf p.46; only Q4_K_M was timed on the board.
- Replace: "which selects the model by comparing candidates and quantisation levels on the same board"
  → "which selects the model by comparing candidates on accuracy at two quantisation levels and on
  latency at \QF{} on the same board".

**I-28 — "Every missed target is … located" is stronger than the results** [FS-I16]
- Location: l.124 (removed if I-01 is applied).
- Evidence: I.pdf p.83 (the recovery remainder is "unexplained").
- Replace: "Every missed target is reported and located." → "Every missed target is reported and
  located as far as the measurement allows."

**I-29 — "Share nothing" is in tension with the measured interference** [NI-14]
- Location: l.248 against l.353–354 and 670.
- Replace: "The two paths share nothing between the audio buffer and the bus." → "The two paths share
  no processing stage between the audio buffer and the bus."

**I-30 — The trajectory figure is small, and the text cites it for all 150 trials** [FT-03]
- Location: l.597, l.612.
- Replace l.612: "All 150 trials converged (Fig.~\ref{fig:trajectories})," → "All 150 trials
  converged (Fig.~\ref{fig:trajectories} shows one trial of each formation),".
- Width: optionally use `\textwidth` at l.597 to lift the text from about 5.3 pt.

**I-31 — Two TikZ labels collide with box borders** [FT-05]
- Location: l.213–214.
- Fix: "not built" → `node[below]`; the "designed / as run" label → `node[left, align=right]`.

**I-32 — Thin segments in the latency figure are unlabelled** [FT-06]
- Location: caption l.476–481.
- Replace: append "Queueing (56~ms) and request overhead with validation (8~ms) are too thin to
  see."

**I-33 — Table 2 overflows, and its reflex row counts are unexplained** [FT-07, BT-06]
- Location: l.445–471.
- Evidence: an overfull box of 3.2 pt; n = 78 of 80 trials; recovery is measured under load only.
- Replace:
  - l.450: `\setlength{\tabcolsep}{2.5pt}` → `{2pt}`.
  - Caption: append "Reflex rows count detections (78 of 80 trials per condition); recovery is
    measured in the loaded condition."

**I-34 — The summary table's false-reject cell lacks intervals and uses an undefined abbreviation** [FT-08, LC-32]
- Location: l.652, l.651.
- Replace: "hold 0.050, abort 0.000 & not dem." → "hold 0.050 [0.001, 0.249], abort 0.000 [0.000, 0.168] & not dem.".
- Caption: change "``Not demonstrated'':" to "``Not dem.'' (not demonstrated):".

**I-35 — The membership-table caption describes the dash backwards** [FT-09]
- Location: l.275–276.
- Replace: "a dash marks the first test, skipped after the second fails." → "a dash marks a latency
  test not applied because the fail-safe test has failed."

**I-36 — The SNR figure plots WER as a fraction, and a label overlaps a target line** [FT-10]
- Fix: in `eval/plots.py` (`article_snr`), plot WER in percent to match Table 4 and offset the 10 dB
  CRR label from the 0.65 line.

**I-37 — The box-plot convention is undefined** [FT-11]
- Location: l.607–609.
- Replace: append "Boxes: interquartile range; line: median; whiskers: 1.5 times the interquartile
  range; circles: outliers."

**I-38 — Lim et al. are described as "zero-shot", which their paper does not state** [CI-11]
- Location: l.141, l.712.
- Evidence: arXiv 2506.07509 (RTX 3080 Ti workstation with Ollama; no few-shot ablation).
- Replace: "prompts open models zero-shot on a local workstation GPU" → "prompts open models on a local
  workstation GPU" (l.141); "prompts open models of the same families zero-shot on a workstation GPU"
  → "prompts open models of the same families on a workstation GPU" (l.712).

**I-39 — Tukey's comparison is cited to the Kramer extension** [CI-10]
- Location: l.423.
- Replace: "with Tukey's comparison~\citep{kramer1956}" → "with the Tukey--Kramer
  comparison~\citep{kramer1956}".

**I-40 — Small language items** [LC-22, 26–29, 31, 33, 34]
- l.349 "The Pi~5's" → "The Raspberry Pi~5's"; Table 2 l.464 and l.467 "Pi compute + publish" →
  "Board compute + publish".
- l.695 "the parser declines almost never" → "the parser almost never abstains".
- l.62–63 "almost entirely before the board's share of 6--18~ms" → "almost entirely upstream of the
  board's own 6--18~ms".
- l.67–68 "which rests on" → "a result that rests on" (covered by I-06).
- l.172–174 "and measure the full speech chain, on recorded audio, on one CPU-only board as the
  condition under which the reflex path must hold" → "and measure the full speech chain on recorded
  audio on one CPU-only board, the condition under which the reflex path must hold".
- l.476 "Where the time goes on each path." → "Latency by stage on each path.".
- l.615 "3500.7" → "3{,}500.7".
- "preempt/preemption" is the American spelling in an otherwise British text; "pre-empt" is
  optional.

### 2.3 Cross-article

**X-01 — The anchor of the end-to-end latency is undefined, and the budget appears to exceed itself** (should-fix) [NM-03, NI-7]
- Location: M l.72–73, l.597–599; I l.64, l.343–346, l.460, l.485, l.680–681.
- Evidence: I.pdf p.51 (Table 4.3: "Parse-path targets start at the end of speech (T0)") and p.52
  ("the clock of the end-to-end target starts when that declaration is made. The wait is therefore
  budgeted as a row of its own … The four that follow T0 sum to 2,600 ms, more than the 2,500 ms they
  sit inside … That target is judged on the measured end-to-end figure").
  - The Ingénieur article lists the 500 ms endpointing wait among the stages of a budget that is
    "2,500~ms end to end from the end of speech". Later it calls 3,602 ms the figure "from the true
    end of speech".
  - The Master article never says where 3,122 ms starts.
- Replace (I l.343–346): "Every stage carries a target at the 95th percentile (p95), fixed before
  any measurement: endpointing wait 500~ms, speech recognition 1,200~ms, …, with 2,500~ms end to end
  from the end of speech." → "Every stage carries a target at the 95th percentile (p95), fixed before
  any measurement. The end-to-end target of 2,500~ms runs from the moment the endpointer declares the
  utterance over; the endpointing wait before it is budgeted separately at 500~ms. After that moment,
  speech recognition is allowed 1,200~ms, language-model prefill 250~ms (assuming a cached prompt
  prefix), decode 1,100~ms, and validation and dispatch 50~ms. These allowances are ceilings that sum
  to 2,600~ms, so the end-to-end target is judged on the measured end-to-end figure, not on the sum of
  stage figures."
- Replace (I l.460): "End to end, from end of speech" → "End to end, from endpointer decision".
- Replace (I l.64): "from the end of speech" → "from the endpointer's end-of-utterance decision".
- Replace (M l.598): "its p95 is 3,122~ms over 226 speech segments," → "its p95, measured from the
  endpointer's end-of-utterance decision, is 3,122~ms over 226 speech segments (3,602~ms when the
  endpointing wait is included),".

**X-02 — The generative-AI declaration lacks the purpose clause of Elsevier's wording** (should-fix) [BT-03]
- Location: M l.725–728; I l.771–774.
- Evidence: Elsevier policy page (fetched): "used [NAME TOOL / SERVICE] in order to [REASON]".
- Replace: "used Claude (Anthropic) for writing assistance." → "used Claude (Anthropic) in order to
  ⟨state the purpose, e.g. draft and edit the text from the authors' theses⟩." **The purpose is
  yours to state**; I will not fill it in without your wording.

**X-03 — The McNemar switch rule is worded differently** (minor) [LC-35]
- Location: M l.328–329 ("from 25 upwards"); I l.424–425 ("above").
- Replace (I): "continuity-corrected~\citep{edwards1948} above" → "continuity-corrected~\citep{edwards1948}
  from 25 upwards".

**X-04 — The `llm2swarm2024` venue is dropped in both bibs** (minor) [CI-03]
- Evidence: arXiv 2410.11387 comment: "Accepted at NeurIPS 2024 Workshop on Open-World Agents";
  the thesis bib keeps it.
- Fix: add `note = {NeurIPS 2024 Workshop on Open-World Agents}` in both bibs.

**X-05 — The 1,658 ms combined figure and its "misses" verdict are not in either thesis** (minor) [FS-X1, NI-10]
- Location: M l.529–530; I l.489–490.
- Evidence: I.pdf p.69 gives no verdict on the combined row, because its allowance assumes a cached
  prefix. The value is in `results/exp2_analysis.md` (1,658, MISSES). It was added by the earlier
  review's fix M-8.
- Replace (M l.530): "which misses the allowance~\citep{companion}" → "above the 1,350~ms allowance,
  which assumes a cached prefix~\citep{companion}".

**X-06 — Terminology and cross-reference style differ between the articles** (minor) [LC-36, LC-37]
- M writes "Figure~\ref" and I writes "Fig.~\ref". M writes "95\% CI" and I writes "95\% interval".
  M writes "abstention" and I writes "declines".
- Fix: optional; align on "Fig." (Elsevier style) and "interval" if desired.

### 2.4 Remaining placeholders (intentional; listed for completeness)

| Item | Master | Ingénieur |
|---|---|---|
| `\cormark[1]` commented out; `\cortext` prints a footnote with no marker | l.40, 46, 54 | l.37, 43, 51 |
| `\ead{<email-author-1>}` / `\ead{<email-author-2>}` | l.41, 47 | l.38, 44 |
| ORCID suppressed (`\renewcommand{\printorcid}{}`), no iDs | l.24–26 | l.21–23 |
| Companion bib entry `note = {Companion paper, in preparation}` | references.bib l.390 | references.bib l.468 |
| AI-declaration purpose (X-02) | resolved, see Fixes applied | resolved |

No TODO, XXX or "??" remains in either source or PDF.

Front-matter checks (pass):
- Highlights: M 79/79/69/77/80 characters, I 81/72/72/78/75.
- Abstracts: 248 and 247 words.
- Keywords: 6 each.
- CRediT, funding, competing interest, data availability and the AI declaration are all present and
  placed before the references. There is no Acknowledgements section (optional).

---

## 3. Comparison with the earlier review (`REVIEW_verification.md`, fixes of 2026-10-13)

### 3.1 Fixes verified in the current `.tex`

All 40 applied fixes are present:
- Master: M-1 to M-17.
- Ingénieur: I-1 to I-13 and I-15 to I-22, I-24 to I-26 (I-14 and I-23 were not applied by design).
- Cross-article: X-1 to X-3.

Spot evidence:

| Earlier ID | Now at | Earlier ID | Now at |
|---|---|---|---|
| M-1 | M l.525, 610 | I-1 | I l.621–623, 706–707 |
| M-2 | M l.528–530 | I-2 | I l.302 |
| M-3 | M l.595 | I-3 | I l.360–364 |
| M-4 | M l.79, 698 | I-4 | I l.309–312, 726–727 |
| M-5 | M l.82, 135–136 | I-5 | I l.507–508 |
| M-6 | Table 1 caption has no path | I-7 | I l.61–62, 743 |
| M-7 | M l.163–164 | I-10 | I l.510–512 |
| M-13 | M l.71–72 | I-11 | I l.366–368 |
| M-15 | M l.122 | I-12 | I l.144 |
| M-17 | M l.267–269 | I-13 | I l.717–719 |
| X-1 | M l.241–244 | I-17 | I l.629–630 |
| X-3 | I l.588 | I-25 | I l.514, 536–537 |

### 3.2 Earlier findings still open

- **I-14 (`henry2026`).** Now confirmable. Crossref resolves the RO-MAN 2026 DOI; see **I-15** above.
- **I-23 (`relays2s`).** Still has no DOI. A venue note is possible now; see **I-16**.
- **M-14 (half-way rounding).** Settled on the thesis's form (0.812), but the opposite-direction
  rounding of 0.9625 → 0.963 remains; see **M-24** (minor, optional).
- **I-7, partly.** The recovery miss is now in the abstract and conclusion. The highlights still
  present preemption without it (highlight 3, l.75), and the abstract figure lacks "at the 95th
  percentile" (**I-20**).

### 3.3 Earlier fixes that introduced new problems, or rest on an unverified premise

- **M-10 (`smollm2` → COLM 2025, "Fully Open" title, OpenReview `3JiCl2A14H`).** This audit could
  not reproduce the confirmation:
  - OpenReview returns a bot challenge.
  - Crossref has no record.
  - A web search found only the arXiv version, whose title lacks "Fully Open".
  - The thesis bib uses the arXiv record.

  Marked UNVERIFIABLE (Appendix B). Either confirm on the COLM 2025 accepted-papers page yourself,
  or revert to the thesis entry (arXiv 2502.02737).
- **M-8 / X-2 (1,658 ms and 1,063 ms added to the Ingénieur text).** Both values are correct
  (`results/exp2_raw/branch_b.jsonl`). However:
  - Neither is in the Ingénieur thesis PDF.
  - The new sentence repeats a "misses" verdict that the thesis deliberately withholds, because the
    1,350 ms allowance assumes a cached prefix.

  See **X-05**.
- **I-2 (4.7 m comparison).** The article is now correct, and the thesis `.tex` was corrected
  (`thesis/ingenieur/ch3_architecture.tex:190`). But the **compiled thesis PDF still says** "larger
  than every admissible inter-drone spacing (1–5 m)" (I.pdf p.48), so the PDF and the `.tex`
  disagree. The PDF is what the jury saw. Recompile the thesis if a corrected version will be
  distributed.
- **I-7 / I-9 (abstract verdicts).** These brought the Ingénieur abstract to 247 words. Any further
  abstract addition (I-06, I-20, I-21) must be paid for by cuts.

### 3.4 Dropped by the earlier review, but confirmed here

- **"60 timed trials instead of 200 planned"** (earlier dropped item 7, "could not confirm a
  200-trial plan"). Confirmed at M.pdf p.46: "The protocol as designed called for 200 repetitions
  per configuration." See **M-04**.

### 3.5 Found by this audit, missed by the earlier one

**Framing and fidelity:**
- I-01 (contributions);
- M-01 (family alone);
- M-02 (best trade-off);
- I-02 (MIRA comparison);
- I-06 (formations not spoken);
- I-22 (class confusion);
- I-26 (guarantee);
- M-20, M-21, M-22.

**Dropped caveats:**
- M-03 (parity scope);
- M-04 (60 of 200);
- M-05 (L2 speaker, accent spread);
- M-06 (CRR from audio);
- I-03 (domain prompt);
- I-05 (state-machine defect);
- I-07 (synthetic voices, stress stream);
- I-17 (acoustic thermal state);
- I-24 (paired-data caveat);
- I-25 (74.2 ms frame).

**Definitions and provenance:**
- X-01 (T0 and budget sum);
- I-12 (850 ms);
- M-11 (abstention);
- M-10 and I-11 (abbreviations);
- I-10 (figure without script);
- M-16 (post-thesis intervals).

**Citations:**
- M-09 (noise and TTS sources);
- M-14 (software);
- M-15 (perplexity);
- I-38 (zero-shot);
- I-39 (Tukey);
- X-04 (LLM2Swarm venue);
- M-34 (DOIs).

**Figures:**
- M-13, M-25, M-28, I-08, I-09, I-30 to I-37.

**Front matter:**
- X-02 (AI declaration wording).

---

## Appendix A. Rejected findings (with reasons)

| Source ID | Claim | Reason rejected |
|---|---|---|
| NM-05, FT-17 | "about 35% or more below the ceiling" is wrong (34.8%) | 34.8% rounds to 35%; "about" covers it. The thesis says "35% clear". The earlier review reached the same conclusion. |
| NM-08, FS-M7, FS-X2, BT-13 (part) | Data availability says "publicly available", but the thesis says private | The GitHub API (2026-10-22) shows the repository public, with release `v1.0.0` carrying the Q4_K_M GGUF. The "every figure" part survives as I-10. |
| NM-10 | The abstract should say that five discordant pairs cannot reach significance | The abstract already says "not significant"; the power argument is in the Discussion (l.618–620). Optional detail, not a defect. |
| FS-M11 | The abstract's ±1.5 needs "exploratory" | The abstract already says "not significant"; the conclusion hedges (l.701). Marginal. |
| FS-M12 | Workstation/Pi identity is claimed for all six | l.293 already says "for the selected configuration". |
| FS-M15 (part) | "One session" is wrong; it was two days | M.pdf p.41: "in one session held over two consecutive days". The article is faithful. The rest survives as M-19. |
| NI-5, FS-I9 | 4.7 m comparison differs from the thesis | The thesis PDF is arithmetically wrong (4.7 < 5); the article is correct. Reported as a PDF/.tex disagreement in §3.3. |
| NI-13 | "Drone identifiers are bounded in length" is wrong | The identifier list is bounded (at most five, M l.217); the wording is accurate. |
| NI-15 | "The field alone let pairs inside 0.80 m 432 times" overstates | I.pdf p.74 uses the same framing ("the potential field alone had failed to keep"). |
| NI-10(c) | "6.5 ms" should be 6 ms | Raw p95 is 6.487 (`results/exp2.csv`); 6.5 is the more precise rounding. The earlier review agreed. |
| FS-I6 | "Requires changing the spotter" over-attributes the 540 ms | I.pdf p.85: "that budget is missed in the spotter's detection, not on the board". The article's Discussion also lists frame quantisation and the anchor (l.672–674). |
| FS-I12 | "The core allocation did what it was designed to do" lacks a control | No frame missed its period, and the article reports the 12 ms load effect in the same paragraph; this matches I.pdf p.67, 85. |
| FS-I15 | The abstract's first sentence is a general claim | It is the design argument the article then states (parse path waits for the endpointer); not unsupported. |
| LC-19 | Decimal places should follow one convention | Four decimals are deliberate for pooled and ablation rates, where the differences are in the third decimal. |
| LC-38 | "8 GB" against "GiB" | 8 GB is the board's marketed size; process memory is measured in GiB. Not an inconsistency. |
| BT-05 | 123.6 pt overfull box at `\maketitle` | Comes from the cas-dc title block (identical in both articles); no visible overrun. |
| BT-07 | Footer "Page 1 of 10" against 11 PDF pages | Template behaviour: the highlights page is unnumbered. |
| BT-09, BT-10, BT-11, BT-14 | Abstract near 250; no graphical abstract; lmodern fallback; optional CRediT roles | Informational, not defects. Re-check line breaks under STIX on Overleaf before submission. |
| FT-19 | Prefill p50 570 against 571 in `exp2_analysis.md` | Raw 570.502; the article matches the thesis. Not an article issue. |
| NI-9 | Keyword training counts (2,100 / 3,930) are not in the thesis | Supported by `data/wake/wake_manifest.json` and by the 70/10/20 split at M.pdf p.42. |

## Appendix B. Unverifiable

| Source ID | Item | Status |
|---|---|---|
| CI-04 (earlier M-10) | `smollm2` as COLM 2025 with the "Fully Open" title and OpenReview `3JiCl2A14H` | OpenReview is behind a bot challenge, Crossref has nothing, and web search finds only arXiv. Needs a manual check. |
| CI-13 | `skysim2026` presented at DAUS 2026 | No proceedings record found. |
| CI-15 | `pyflyt` still unpublished | No published version found; absence is not proven. |
| CI-16 | `dregon` URL | Did not respond; DOI and metadata are correct. |
| CI-06 | `awq` pages | Not shown on the MLSys page. |
| — | `landau2017` "regular expressions" claim (I l.137) | Abstract paywalled; rests on the thesis statement. |
| X-02 | Purpose of AI use in writing | Only the authors can state it. |

## Appendix C. Thesis-internal issues noted in passing (no article change)

- I.pdf p.73 says the Master thesis "re-baselined the criterion [safe-failure] to … 0.053". The
  Master thesis instead says neither abstention target is adjusted (M.pdf p.56). The articles follow
  the Master thesis.
- I.pdf p.75 (Table 6.3, end-to-end command-to-formation latency over a "Zenoh bus", reflex stop at
  0.30 s) contradicts the thesis's own measurements (545 ms reflex p95, UDP over loopback, link to
  the controller not built). The articles correctly leave it out.
- I.pdf p.48 still has the 4.7 m error (see §3.3).

---

## Fixes applied (2026-10-22)

**Scope approved:** all errors and should-fixes, plus the minors that concern accuracy, fidelity,
citations or misleading captions.

**Change marking.** Every edit is marked in the source with the `changes` package. Each article's
preamble has one switch:

```
\usepackage[draft,commandnameprefix=ifneeded]{changes}   % switch to [final] to strip all markup before submission
```

`cas-dc` already defines `\comment`, so `commandnameprefix=ifneeded` keeps `\added`, `\deleted`
and `\replaced` and renames only the margin note to `\chcomment`.

The marks are:
- **Insertions** are blue.
- **Deletions** are red and struck through.
- **Margin notes** carry the finding ID.
- **Inside the abstract, highlights and floats**, where margin notes are impossible (cas-dc writes
  the abstract to a file, and a margin note inside a float is lost), the ID is printed inline as a
  small blue `[ID]` (`\AuditID`).
- **The Ingénieur contributions list (I-01)** is a whole environment, which cannot be struck
  through. It is shown in red (old) and blue (new) through `\AuditDeletedBlock` /
  `\AuditAddedBlock`.

All of this markup disappears under `[final]`.

**Verification:**
- Both articles compile under `[draft]` and `[final]` with 0 LaTeX errors and 0 undefined
  references or citations (Master 12 pages, Ingénieur 13 pages).
- The final PDFs contain no markup.
- The only overfull box left in the final builds is the template's `\maketitle` box (Appendix A,
  BT-05). The Ingénieur latency table no longer overflows (I-33).
- In draft mode the latency table overflows because its relabelled row shows both versions.
- One build note: a `changes` argument that begins with `~` breaks `[final]`, so the three new
  citations in M-14 keep `~` outside the markup.

### Master (`articles/master/main.tex`)

| ID | Change |
|---|---|
| M-01 | Abstract, highlight 3 and contribution 2: "family alone" → the two families differ, marked exploratory, with prompt rendering and training session named, under one training budget. |
| M-02 | Conclusion: "offers the best accuracy–efficiency trade-off" → "is the configuration the selection rule retains"; adds that SmolLM2-360M and Qwen2.5-0.5B are the two non-dominated points, selected between on accuracy. |
| M-03 | Parity check scoped to Q4_K_M; Q8_0 covered by identical metadata only; Llama-3.2 date difference stated. |
| M-04 | Timing protocol notes that 200 repetitions were planned; Limitations adds the 60-repetition caveat on the narrow decode margin. |
| M-05 | Limitations: the golden speaker is non-native; Common Voice accent-group WER 6.2–49.5%; transfer to other speakers not established. |
| M-06 | Limitations: CRR 0.690 clean and 0.590 at 10 dB against 0.935 EM, cited to the companion paper. |
| M-07 | Memory figures named per model. |
| M-08 | "More than 40% of the span" → "of the same order as the 15.0 pp span …, whose endpoints also differ in family". |
| M-09 | Training audio: Piper TTS, DREGON and ESC-50 named and cited. |
| M-10 | LoRA, GGUF, GBNF, CI and NLP expanded at first body use. |
| M-11 | Abstention defined in Metrics. |
| M-12 | Throughput definition rewritten. |
| M-13 | `fig_thermal.pdf` regenerated. A value label that the target line would strike through is now drawn inside its bar, and the allowance label is two lines so it clears the Llama bars (`eval/plots.py`, `_paired_panels` and `article_thermal`). `fig_ood.pdf`, which shares the code, regenerates byte-identical. The caption carries `[M-13]`. |
| M-14 | `peft`, `transformers` and PyTorch cited. |
| M-15 | Perplexity attributed to the `llama.cpp` formats only. |
| M-17 | Contribution 4: "including" → "beside". |
| M-18 | Limitations: only two GGUF quantisation levels examined. |
| M-19 | Golden set: the `set_param` pattern with altitude and spacing never occurs in training. |
| M-20 | "Mattered as much as having one" → "also mattered". |
| M-21 | Discussion heading scoped to the golden set. |
| M-22 | LLM2Swarm sentence: "or test a small model for generation speed only". |
| M-23 | Preliminary test: five commands, two looped, resolved to a hover, landing not executed. |
| M-26 | Table 1 caption: bold explained. |
| M-27 | Table 2 caption: Δ = deployed minus fp16. |
| M-28 | Fig. 6 caption: point labels defined. The in-figure "stage allowance" label is deferred (plot change). |
| X-01 | The 3,122 ms is said to start at the endpointer's decision; 3,602 ms with the wait included. |
| X-05 | "Which misses the allowance" → "above the 1,350 ms allowance, which assumes a cached prefix". |
| X-02 | AI declaration: "for writing assistance" → "in order to assist with the writing of the manuscript", and the second sentence shortened to "The authors reviewed the content and take full responsibility for the publication." (wording chosen by the authors, 2026-10-22; it keeps the responsibility statement of Elsevier's template but not its exact text). |

### Ingénieur (`articles/ingenieur/main.tex`)

| ID | Change |
|---|---|
| I-01 | The contributions list is replaced: the claim is the membership rule and the reflex path's own latency budget. The runtime, the ordering rule and the validation chain are described as engineering, preemption is credited as prior art (MIRA), and the evaluation is described as what the article reports. |
| I-02 | Discussion: MIRA's 466 ms median from speech onset against our 965 ms median from keyword onset; what the reflex adds is a derived rule and a budget, not speed. |
| I-03 | whisper.cpp's initial prompt listing the command vocabulary is stated. |
| I-04 | Failure causes: one split utterance is counted under the parser-error cause, which is assigned first. |
| I-05 | Limitations: the state-machine legality defect is stated. |
| I-06 | Abstract: formations were commanded directly, not by voice, in a kinematic simulator. |
| I-07 | The stress stream is "not an operating rate"; the synthetic takes use the three training voices. |
| I-08 | Keyword-curve caption: the false-reject criterion is judged on the recorded takes. |
| I-09 | Summary table: "p50/p95/p99" → "p50 and p95". |
| I-10 | Data availability excludes Fig. 5, which is distributed as rendered; Fig. 5 caption gives trial 1 (seed 42) and the colour meaning. |
| I-11 | VAD, GBNF, UDP, PID, ANOVA, p50 and p99 expanded or defined. |
| I-12 | The 850 ms onset target is defined in the budget section. |
| I-13 | The ill-formed sentence on the acoustic model is rewritten. |
| I-14 | The dangling "which the endpointer split" is fixed. |
| I-17 | Acoustic experiment: 54.3–80.1 °C; soft temperature-limit flag set during 5 dB. |
| I-18 | Temperature "sampled at the start of each trial". |
| I-19 | "Takes" → "trials" for the 40 abort and 40 hold replays. |
| I-20 | Abstract: recovery "at the 95th percentile". |
| I-22 | "Never confused on the recorded takes". |
| I-23 | Conclusion: 6–18 ms "before the stop was published on the bus". |
| I-24 | The paired tests are named as the primary analysis. |
| I-25 | Longest frame of 74.2 ms while speech recognition ran. |
| I-26 | "Guarantee" → "mechanism". |
| I-27 | The companion paper's selection: accuracy at two quantisation levels, latency at Q4_K_M. |
| I-29 | "Share nothing" → "share no processing stage". |
| I-30 | Fig. 5 is cited as showing one trial of each formation. |
| I-33 | Table 2: `\tabcolsep` 2.5 pt → 2 pt (not colourable); caption explains n = 78 and that recovery is measured in the loaded condition. |
| I-34 | Summary table: false-reject intervals added; caption defines "not dem.". |
| I-35 | Membership-table caption: dash wording fixed. |
| I-37 | Box-plot convention defined in the caption. |
| I-38 | "Zero-shot" removed for Lim et al. (two places). |
| I-39 | "Tukey's comparison" → "the Tukey–Kramer comparison" in Methods. |
| X-01 | Budget section rewritten around the endpointer's decision, with the 2,600 ms sum explained. The table row reads "End to end, after endpointing", and the abstract and §4.1 say "from the endpointer's end-of-utterance decision". |
| X-03 | McNemar switch: "from 25 upwards". |
| X-02 | AI declaration: same wording as in the Master article. |

### Bibliography changes (not colourable in the PDF)

- `articles/master/references.bib`:
  - added `piper`, `dregon`, `esc50`, `commonvoice`, `transformers`, `peft`, `pytorch`, copied from
    the thesis and Ingénieur bibs, with annotation notes removed and `urldate` written as an
    "Accessed" note as elsewhere (M-05, M-09, M-14);
  - `geng2023` gains `doi = {10.18653/v1/2023.emnlp-main.674}` and `qin2021` gains
    `doi = {10.24963/ijcai.2021/622}`, both re-resolved on Crossref (M-34);
  - `llm2swarm2024` gains `note = {NeurIPS 2024 Workshop on Open-World Agents}` (X-04).
- `articles/ingenieur/references.bib`:
  - `henry2026` is now `@inproceedings` at RO-MAN 2026, pp. 2455–2460,
    DOI 10.1109/RO-MAN72286.2026.11714651 (I-15);
  - `relays2s` gains `note = {Accepted to Findings of EMNLP 2026}`, from the arXiv comment
    re-checked on 2026-10-22 (I-16);
  - `llm2swarm2024` gains the same venue note as in the Master bib (X-04).
  - `smollm2` reverted from the unverified COLM 2025 entry ("Fully Open" title, OpenReview
    `3JiCl2A14H`) to the thesis's arXiv record (arXiv 2502.02737, whose title has no "Fully Open");
    see §3.3 and Appendix B (CI-04). Applied 2026-10-22 at the authors' request.
- BibTeX warns on 7 Master entries without pages (`pytorch` added; `smollm2` is now `@misc` and no
  longer warns); the Ingénieur bib has none.

### Not applied

- **Deferred plot changes:** M-25, the in-figure label of M-28, I-31, I-32 and I-36.
- **Style items:**
  - M-24, M-29, M-30, M-31, M-32 and M-33;
  - I-21 and I-40;
  - X-06;
  - M-16 (no text change).
- **I-28.** Superseded by I-01: the sentence it targeted is gone.

### Consequences to note

- **Abstract length.** The approved abstract additions (M-01, I-06, I-20, X-01) bring the abstracts
  to about 258 words (Master) and 262 (Ingénieur), from 248 and 247. If the target journal caps
  abstracts at 250, cuts are needed. Candidates:
  - Master: the clause "from the fp16 model to the quantised artefact under the grammar";
  - Ingénieur: the opening sentence, or "with no class confusion".
- **Page count.** Master 11 → 12 pages, Ingénieur 12 → 13 (final mode).
