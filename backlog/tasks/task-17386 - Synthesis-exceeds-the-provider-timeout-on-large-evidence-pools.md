---
id: TASK-17386
title: Synthesis exceeds the provider timeout on large evidence pools
status: To Do
assignee: []
created_date: '2026-08-17 16:45'
labels:
  - research
  - websearch
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A research run whose evidence pool is large enough can lose its entire report to the provider's request timeout during final synthesis. The run collects and judges its sources normally, the gate admits them, and then the single synthesis call never returns within the configured wall clock, so the run produces no answer and no verification payload at all — the work spent on collection and judgement is discarded.

This now matters much more than when it was harmless. Multi-hop ships enabled by default and sub-question fan-out reaches the academic providers, so evidence pools are substantially larger than the ones every earlier measurement used; and per-chunk summarization now succeeds rather than failing fast, which lengthens the reduce step it feeds. The failure is silent in the metrics, because a run that produces no payload is simply absent from the aggregate rather than scored as a failure.

Bounding the synthesis input, streaming it, or deriving the timeout from the pool size are all plausible answers; deciding between them needs the measurement below rather than taste.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A run whose synthesis cannot complete within the wall clock produces a recorded, legible terminal state instead of vanishing from results
- [ ] #2 The relationship between evidence-pool size and synthesis wall clock is measured, not assumed
- [ ] #3 A run on a local model with a pool the size of a default multi-hop run completes its synthesis, or is bounded so that it can
- [ ] #4 Whatever bound is chosen is stated where a user meets it, alongside the existing decomposition spend notes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Observed in task-17384's arm H (2026-08-17), the first arm carrying both the
academic fan-out fix (task-17372) and the chunk-summarization fix:

- Two of three questions scored normally and well: markers 23/23 and 32/32,
  `cited_sentence_ratio` 1.00 on both, zero chunk failures.
- The third produced nothing. Its log ends in
  `urllib3.exceptions.ReadTimeoutError: HTTPConnectionPool(host='127.0.0.1',
  port=9191): Read timed out. (read timeout=600)` on `/v1/chat/completions`,
  followed by the recorder's `[no citation verification on the synthesis
  branch]`.
- That run's pool was the largest of the three: the arm admitted 66 sources
  across its questions, against 50 in the arm before the fan-out fix and 32
  before that.
- 600s is what `record_research_baseline._prime_local_llm_url` sets as
  `api_timeout`; the shipped provider default is lower, so an ordinary run is
  more exposed than this measurement was.
<!-- SECTION:NOTES:END -->
