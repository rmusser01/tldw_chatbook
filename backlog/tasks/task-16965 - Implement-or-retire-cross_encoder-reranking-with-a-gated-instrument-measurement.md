---
id: TASK-16965
title: >-
  Implement or retire cross_encoder reranking, with a gated-instrument
  measurement
status: In Progress
assignee: []
created_date: '2026-08-16'
updated_date: '2026-08-17 21:00'
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
- [ ] #1 A recorded decision exists for `cross_encoder`: implemented as a local deterministic reranking strategy, or retired from the strategy vocabulary -- not left as an unimplemented name
- [ ] #2 If implemented: its retrieval effect is measured on the gated instrument as a pre-registered before/after over the 105 metrics, with the rule for "this helped" fixed BEFORE the run
- [ ] #3 The measurement is decisive in both directions: a null result (no census gain, no cell moved beyond the gate's tolerance) is reported as the answer and is sufficient grounds to retire the strategy
- [ ] #4 If retired (or declined): `config_profiles.py`'s not-implemented comment and any user-facing strategy vocabulary stop implying `cross_encoder` is forthcoming, and Hybrid Full's `pointwise` substitution is documented as the permanent choice rather than a stopgap
- [ ] #5 The measurement requires no live provider spend and no network -- the whole reason a local cross-encoder is the measurable one
- [ ] #6 The gate still reads `PASSED: No regression. 105 metric(s)` on the shipped state, whichever arm is taken
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
T1: implement CrossEncoderReranker (local, credential-free) behind create_reranker; RED tests with a stub model (no download).\nT2: env-gated offline probe over semantic+hybrid, prints census + VERDICT.\nT3: obey the pre-registered verdict (ship-with-docs or retire the name) and close.\nDecision rule fixed in Docs/superpowers/plans/2026-08-17-cross-encoder-measurement.md BEFORE any run.
<!-- SECTION:PLAN:END -->
