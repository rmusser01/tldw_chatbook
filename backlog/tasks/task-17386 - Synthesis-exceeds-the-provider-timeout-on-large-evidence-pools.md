---
id: TASK-17386
title: Synthesis exceeds the provider timeout on large evidence pools
status: In Progress
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
- [x] #1 A run whose synthesis cannot complete within the wall clock produces a recorded, legible terminal state instead of vanishing from results
- [x] #2 The relationship between evidence-pool size and synthesis wall clock is measured, not assumed
- [ ] #3 A run on a local model with a pool the size of a default multi-hop run completes its synthesis, or is bounded so that it can
- [x] #4 Whatever bound is chosen is stated where a user meets it, alongside the existing decomposition spend notes
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

## Implementation Notes (partial -- AC #3 remains open)

<!-- SECTION:NOTES:BEGIN -->
AC #1, #2 and #4 are closed; **AC #3 is deliberately NOT closed** and the task
stays open for it.

The failure is now legible. `aggregate_results` records the failure CLASS --
never its message, since a timeout's text carries host:port and this value
travels into a run's artifacts -- with the pool size it failed on, and the
engine appends the warning while the round's warning list is still being built,
so the reason reaches the run's warnings and bundle rather than only a summary
written later. `FinalAnswerDict` declares the field rather than growing an
undocumented key. Before this, such a run completed with a generic string,
carried no citation verdict, and was simply absent from any aggregate: the
failure removed itself from the sample, so metrics over surviving runs looked
better for it.

AC #2, measured from the recorded arms rather than assumed: pools of 14-32
sources synthesized in 185-328s, while pools of 46-66 hit 1200s (two 600s
attempts, `MaxRetryError`) and 970s (a timed-out attempt plus a successful
retry). Synthesis on a default multi-hop pool routinely exceeds a 600s
per-attempt budget, so this is the normal shape of a large-pool run rather than
an exotic error.

AC #3 needs a size-aware synthesis budget, and the per-call timeout is not
plumbable through `chat_api_call` -- its signature is shared by roughly nine
providers, so threading one through is a change to the shared dispatcher rather
than to this pipeline, and belongs in its own task. Until it exists, the
documented answer is a provider `api_timeout` well above 600s for local models,
recorded beside the measurement in the eval baseline doc.
<!-- SECTION:NOTES:END -->
