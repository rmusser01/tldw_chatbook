---
id: TASK-16965
title: >-
  Implement or retire cross_encoder reranking, with a gated-instrument
  measurement
status: Done
assignee: []
created_date: '2026-08-16'
updated_date: '2026-08-17 22:15'
labels:
  - rag
  - measurement
dependencies:
  - TASK-3502
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reranking's retrieval VALUE in this repo has never been measured, and TASK-3502
declared that measurement explicitly out of scope with a reason: the three
implemented reranking strategies (pointwise/pairwise/listwise) are all
LLM-driven, and an LLM reranker cannot be measured on the gated instrument --
the instrument (`Tests/RAG_Eval/`, 105 metrics) is local and deterministic
while the reranker is remote, priced and non-reproducible. So TASK-3502 made
the reranker HONEST (provider choice, cost disclosure, degradation surfaced,
no "| reranked" over-claim) without ever answering whether reranking helps.

A local cross-encoder CAN be measured there: deterministic, no spend, runs
inside the gate. It is also the strategy the repo already gestures at and does
not have. `RAG_Search/config_profiles.py:346-352` says so in a standing
comment: `"cross_encoder" is not an implemented reranking strategy in chatbook
-- reranker.py only implements the three LLM-driven strategies
(pointwise/pairwise/listwise); there is no local cross-encoder model path.
This profile previously requested "cross_encoder" and raised ValueError the
moment its reranker tried to construct (task-3170 P0).` The Hybrid Full
profile now ships `pointwise` as the nearest substitute -- a stopgap that has
outlived its incident and that nobody has ever measured.

This task owns the question TASK-3502 could not answer: implement
`cross_encoder` as a local deterministic strategy and MEASURE it on the gated
instrument, or retire the name. **Retire is pre-registered as an acceptable
outcome**: if the measurement shows nothing -- no census gain, no cell moved
beyond noise -- that is a publishable answer (the RAG programme has already
shipped one pure null arc, PRF/TASK-15965) and it licenses deleting the
strategy name, the stopgap comment and the expectation, rather than leaving a
half-promised feature in the vocabulary. Choosing NOT to implement, on the
grounds that a local cross-encoder model dependency is not worth its weight,
is likewise an acceptable outcome provided it is recorded as a decision.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A recorded decision exists for `cross_encoder`: implemented as a local deterministic reranking strategy, or retired from the strategy vocabulary -- not left as an unimplemented name
- [x] #2 If implemented: its retrieval effect is measured on the gated instrument as a pre-registered before/after over the 105 metrics, with the rule for "this helped" fixed BEFORE the run
- [x] #3 The measurement is decisive in both directions: a null result (no census gain, no cell moved beyond the gate's tolerance) is reported as the answer and is sufficient grounds to retire the strategy
- [x] #4 If retired (or declined): `config_profiles.py`'s not-implemented comment and any user-facing strategy vocabulary stop implying `cross_encoder` is forthcoming, and Hybrid Full's `pointwise` substitution is documented as the permanent choice rather than a stopgap
- [x] #5 The measurement requires no live provider spend and no network -- the whole reason a local cross-encoder is the measurable one
- [x] #6 The gate still reads `PASSED: No regression. 105 metric(s)` on the shipped state, whichever arm is taken
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
T1: implement CrossEncoderReranker (local, credential-free) behind create_reranker; RED tests with a stub model (no download).\nT2: env-gated offline probe over semantic+hybrid, prints census + VERDICT.\nT3: obey the pre-registered verdict (ship-with-docs or retire the name) and close.\nDecision rule fixed in Docs/superpowers/plans/2026-08-17-cross-encoder-measurement.md BEFORE any run.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented `cross_encoder` as a local, credential-free reranking strategy, MEASURED it on the gated eval instrument against a rule fixed before the code was written, and then acted on the measurement. The measured answer was that it does not help. Per the owner's ruling below, the code stays and the PROMISE is retired: `CrossEncoderReranker` remains implemented and selectable, every claim or implication that it improves search is gone, and the measured harm now travels with it everywhere the strategy is nameable.

**THE OWNER RULING (2026-08-17), asked explicitly with the trade-offs on the table: "KEEP THE CODE, RETIRE THE PROMISE."** This overrides the plan's arm B (full retirement of the name). The reason it was worth asking rather than retiring silently is the bimodal finding below: the strategy is a large win exactly where retrieval is weak, so deleting it would have deleted the only reranking path this repo can measure at all.

## The census (measured before any decision, and the reason the arc proceeded)

Reproduced by the probe from its OWN retrievals, not quoted: `plain` returns >=2 rows on **0/60** queries (39 return 0, 21 exactly 1) -- reranking is provably the identity there, so `plain` ran as a STOP guard and moved nothing (0 rows moved, every delta exactly 0.000). `semantic` and `hybrid` both return a FULL window on 60/60 at k=10 and k=20. So a null could not have been dismissed as "nothing to reorder".

## The pre-registered rule (quoted, and fixed BEFORE the strategy existed)

From `Docs/superpowers/plans/2026-08-17-cross-encoder-measurement.md`, written before Task 1:

> **HELPED** -- at least one of MRR/NDCG/P@k improves beyond the gate's tolerance (0.05) on at least one mode, AND no category regresses beyond tolerance. [...] **NULL** -- nothing moves beyond tolerance on either mode. -> report as the answer and RETIRE the name. **HARMED** -- any regression beyond tolerance. -> retire, and say so plainly. [...] Ties, partial movement, or a mixed picture resolve to NULL -- the burden is on the strategy to show a gain.

It was also implemented as a TESTED pure function (`cross_encoder_probe.arm_verdict` / `compose_arc_verdict`, 19 always-on tests) importing the gate's own `FAIL_BAND` rather than copying `0.05`, so it could not be edited after the fact without a reviewable diff.

## Two arms, both declared before the run

Arm A reranks the k=10 window. Because `precision_at_k`/`recall_at_k`/`f1_at_k` are set functions of `retrieved_ids[:k]`, permuting a <=k list cannot move them -- the rule's own P@k clause is VACUOUS on arm A alone. Arm B therefore retrieves 20, reranks, and scores the first 10; it is the only arm where P/R/F1 are live. `compose_arc_verdict` fixes that harm in one arm is never covered by help in the other.

Overall row, @k=10:

| arm | mode | MRR | NDCG | R@k |
|---|---|---|---|---|
| A | semantic | 0.804 -> 0.772 | 0.804 -> 0.780 | invariant by construction |
| A | hybrid | 0.809 -> 0.798 | 0.817 -> 0.810 | invariant by construction |
| B | semantic | 0.808 -> 0.762 | 0.804 -> 0.776 | 0.804 -> 0.826 |
| B | hybrid | 0.812 -> 0.787 | 0.817 -> 0.805 | 0.848 -> 0.870 |

## VERDICT: HARMED

Nothing moves beyond the 0.05 band on the OVERALL row in either arm; the verdict comes from the rule's CATEGORY clause -- `paraphrase` MRR 1.000 -> 0.923 (A) / 0.872 (B) and NDCG 1.000 -> 0.943 / 0.905, `vocabulary_mismatch` MRR 1.000 -> 0.944, on both modes and both arms. **Disclosed reading ambiguity:** the rule writes its gain clause at MODE level and its regression clause at CATEGORY level, which is what the probe implements; read on the overall row ALONE the verdict is NULL. Both readings say "do not recommend this", which is what shipped.

## The bimodal finding (why the owner was asked, not told)

The strategy is NOT inert -- 3,621 rows scored, 0 failed, 1,950 rows moved, 17.2s inside `rerank()`. Its effect splits by query category. Large gains where retrieval was weak: hybrid `scoped` MRR **0.163 -> 0.929** (NDCG 0.348 -> 0.947), hybrid `prompt` 0.022 -> 0.200, semantic `negation` NDCG 0.000 -> 0.105. Losses where retrieval was already perfect: `paraphrase` (13 q) and `vocabulary_mismatch` (9 q) both sat at MRR 1.000, so the only movement available was down, and four queries lost rank 1 (`pr-requisition-freeze`, `pr-photovoltaic-roof`, `vm-nearsightedness`, `pr-standup-slot`). On the averaged row the two nearly cancel: arm B buys R@10 +0.022 and pays MRR -0.046/-0.026.

## Acceptance criteria, against evidence

- **#1 (recorded decision).** IMPLEMENTED and kept: `CrossEncoderReranker` in `RAG_Search/reranker.py`, dispatched by `create_reranker_from_config`, in the `RerankingConfig.strategy` Literal. Not left as an unimplemented name in either direction. Owner ruling recorded above and in the class docstring.
- **#2 (measured, rule fixed first).** `Tests/RAG_Eval/harness/cross_encoder_probe.py` + `test_cross_encoder_probe_run.py`; the rule was written into the plan before Task 1 and encoded as tested code before the run produced a number. Instrument pinned, never the outcome: arm A's before-column equals `run_eval`'s own three-mode report to 1e-9 on all 15 cells, every pass asserted on its expected backend, `plain` asserted immobile.
- **#3 (decisive in both directions).** The arc was pre-authorised to publish a null and retire on it. It returned HARMED, reported plainly, and the strategy is now the default and recommendation of nothing. Two traps that would have destroyed decisiveness were caught: a fabricated NULL (below) and a float-noise exact-tie reading as HELPED (`0.5 + 0.05 - 0.5 == 0.05000000000000004`), guarded with a 1e-9 epsilon so a draw resolves to NULL.
- **#4 (vocabulary stops implying it is forthcoming; Hybrid Full documented as permanent).** `config_profiles.py`'s not-implemented comment is rewritten: it now records that `cross_encoder` IS implemented, carries the verdict, and states that Hybrid Full keeps `pointwise` as a **permanent, measured** choice rather than a stopgap -- the measurement forbids recommending the switch. Every other user-visible surface now states what was measured: `reranker.py` module + class docstrings, the `strategy` Literal, `config.py`'s config template, `Helper_Scripts/rag_config_examples/rag_v2_example.toml`, `Config_Files/rag_pipelines.toml`, `Docs/User_Guide/settings/rag.md` (new strategy-table row + a verdict quirk with the before/after table + a Verified-against stamp), `Docs/Development/RAG/RAG-DESIGN.md` (it was listed under **Future Enhancements** -- now struck, done and measured), `Docs/Development/RAG/RAG-Documentation.md` ("Improve result quality with re-ranking" was a claim; corrected), and `Tools/document_expansion_tool.py` (its docstring asserted `cross_encoder` was unimplemented -- now false).
- **#5 (no spend, no network).** Confirmed three ways in the run: `Tests/conftest.py`'s autouse `_no_network_io` socket guard recorded zero blocked attempts, the run asserts `huggingface_hub.constants.HF_HUB_OFFLINE is True`, and `chat_api_call` -- the seam the other three strategies bill through -- was monkeypatched to RAISE for the duration and was never called. The implementation reads no credential and imports no provider path (asserted by AST, not grep: a docstring mention had defeated the substring check). 25.7s wall clock, entirely local.
- **#6 (gate green on the shipped state).** Re-run on the final tree: `[rag-eval baselines] PASSED: No regression. 105 metric(s) within 0.05 of baseline.` No baseline was re-stamped -- nothing this arc shipped changes retrieval, by design.

## The best catch of the arc: a near-fabricated null

The measurement was ONE monkeypatch away from a false 0.000. `Tests/conftest.py` sandboxes `HOME`, and `huggingface_hub.constants.HF_HUB_CACHE` is computed from `expanduser("~")` AT IMPORT -- so under pytest `CrossEncoder(...)` raises `OSError` on a machine where the model IS cached. `CrossEncoderReranker` DEGRADES rather than raises (the TASK-3502 contract), so every window would have come back unchanged, every metric would have been graded on un-reranked output, and the before/after table would have read a clean 0.000 delta on all 105 cells: a null that looked perfect, was pre-authorised as an acceptable outcome, and was entirely manufactured. Unlike a fallback that makes a number look GOOD, this one has no tell at all -- a real null and a never-ran null are the same table. The run repoints the constant and asserts `rows_scored > 0` and `rows_failed == 0` per pass. Lesson recorded by EXTENDING `backlog/docs/lessons-testing-evidence.md`'s "A metric can be graded on fallback content" entry (same family, sharper rule for A/B deltas) plus a cross-reference from the `HF_HUB_OFFLINE` entry, rather than adding a fourth near-duplicate.

## Files

Implementation: `tldw_chatbook/RAG_Search/reranker.py`. Probe/record: `Tests/RAG_Eval/harness/cross_encoder_probe.py`, `Tests/RAG_Eval/test_cross_encoder_probe.py`, `Tests/RAG_Eval/test_cross_encoder_probe_run.py`, `Tests/RAG_Search/test_cross_encoder_reranker.py`, `Docs/superpowers/qa/2026-08-17-cross-encoder/report.md`. Promise retirement: `tldw_chatbook/RAG_Search/config_profiles.py`, `tldw_chatbook/config.py`, `tldw_chatbook/Config_Files/rag_pipelines.toml`, `tldw_chatbook/Tools/document_expansion_tool.py`, `Helper_Scripts/rag_config_examples/rag_v2_example.toml`, `Docs/User_Guide/settings/rag.md`, `Docs/Development/RAG/RAG-DESIGN.md`, `Docs/Development/RAG/RAG-Documentation.md`, `backlog/docs/lessons-testing-evidence.md`.

Left alone deliberately: `Library/library_rag_score_kinds.py` and `Library/library_rag_state.py` mention cross-encoder logits as a SCORE SCALE (factual, no promise); `UI/Tools_Settings_Window.py`'s `cross-encoder/ms-marco-MiniLM-L-12-v2` default is a reranker MODEL name on a nav-unreachable deprecated window (TASK-1346), not strategy vocabulary.

Batteries: `Tests/RAG_Search/` + `Tests/RAG_Eval/` = 678 passed / 26 skipped. Repo-wide collection 49,427 (2 pre-existing `playwright` ImportErrors in `Tests/Web_Scraping/Confluence/`, untouched by this branch). ruff clean on every touched file; `ruff format --check` deltas on `config_profiles.py`/`config.py`/`document_expansion_tool.py` are pre-existing at HEAD, `reranker.py` is formatted.
<!-- SECTION:NOTES:END -->

### Final-review corrections (2026-08-17, applied before merge)

- **F1 — the headline was narrower than it sounded.** "Net harmful on the
  averaged row" used the instrument's `overall`, which EXCLUDES `scoped` and
  `negative` (`UNAVERAGED_CATEGORIES`) — and `scoped` is exactly where this
  strategy wins. Averaged over all 53 ground-truthed queries, **hybrid
  reverses sign: MRR 0.731 → 0.806 (+0.075)**. The pre-registered verdict is
  unchanged (it is computed from the instrument's own cells, and the six
  regression cells stand), but every headline site now carries the caveat.
- **F2** — arm B's MRR is unbounded-rank, not MRR@10: its reranked lists
  exceed 10 on 60/60 queries (mean 19.4 semantic / 20.0 hybrid). Correcting
  to @10 moves the deltas slightly (semantic −0.0169 → −0.0134) and does not
  change the verdict; the label is corrected and the caveat recorded.
- **F4** — `library_rag_score_kinds.py` was modified in Task 1 and is now in
  the files list (the notes had listed it as untouched).
- **F5 — the normalisation claim was overstated, and the control now backs
  the corrected version.** Ordering is normalisation-invariant only when
  `combine_original_score` is False; the shipped config blends 30% of the
  original score, so the sort key is scale-dependent. The review RAN that
  control: with the blend off (ordering identical to raw logits) the verdict
  is still HARMED with the same six regression cells and 3dp-matching deltas,
  and a 10×-scaled blend agrees. **The harm is the model's ranking, not the
  min-max.**
- **F3** (`reranking_strategy` has zero readers repo-wide) folded into
  TASK-17600's scope — it is the same enabled-but-unread species that task
  already owns.
- **F6** (the work guard asserts rows scored/failed but not
  `row_order_changes > 0`) recorded in the QA report as a known limit of the
  guard; the mutation test proves the guard catches the failure mode that
  actually occurred.
