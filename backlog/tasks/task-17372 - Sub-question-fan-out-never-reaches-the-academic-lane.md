---
id: TASK-17372
title: Sub-question fan-out never reaches the academic lane
status: In Progress
assignee:
  - '@robert'
created_date: '2026-08-17 07:35'
labels:
  - research
  - web-tools
dependencies:
  - task-17370
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase-1 sub-question generation happens inside the web pipeline, so the generated sub-questions only ever drive web searches. The local research engine's first round collects with `round_queries = [question]`, and the academic lane loops over exactly those queries — so for a papers/repositories run, enabling fan-out adds no new academic evidence at all. The sub-questions do reach the relevance gate (the merged pool is analyzed with the accumulated sub-question list), so fan-out changes how academic evidence is JUDGED while leaving what is RETRIEVED untouched. Retrieval-side decomposition currently arrives only via gap-driven replanning, whose later rounds do search their gap questions. Decide whether the academic lane should search the sub-questions too, and make the asymmetry deliberate rather than incidental.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Whether the academic lane searches generated sub-questions is a stated, tested decision rather than a side effect of where sub-question generation lives.
- [x] #2 If the lane does fan out, its extra searches are counted against the same budget ledger and search cap as the web lane's, with no path that spends past the cap.
- [x] #3 The gate-versus-retrieval asymmetry is documented wherever the decomposition settings are described, so a measured result cannot be misread as evidence about retrieval.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Decide the asymmetry deliberately: the lane searches generated facets, since
   without that fan-out cannot affect retrieval at all and its only measured
   effect (gate context) is nil.
2. Build the lane's query list in one place -- primary queries plus facets,
   deduplicated case-insensitively.
3. Bound it twice: the same query cap the web lane obeys, and a ledger
   reservation per extra query so a tight max_searches cannot be exceeded.
4. Cover the decision, the cap, the budget and the dedup with tests.
5. Document the change where the decomposition settings are described, and scope
   the already-recorded fan-out measurement to what it actually measured.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The academic lane now searches the generated sub-questions, via a single
`_academic_queries` helper that builds the round's query list. The decision is
stated rather than incidental: fan-out's only measured effect was on the gate
(task-17370 recorded 0.42 -> 0.38, i.e. flat), so unless the facets reach
retrieval the feature has no demonstrated purpose.

Bounded twice over. The total is capped by the same
`search_default_max_queries` the web lane obeys, and each EXTRA query reserves
and settles one search against the ledger, so a tight `max_searches` cannot be
exceeded by the lane fanning out. The base `round_queries` keep today's
accounting (uncounted) deliberately -- counting them would silently shrink every
existing run's web budget, which is a separate decision from this one and is
recorded as such in the helper's docstring.

Dedup is case-insensitive and whitespace-trimmed, because the pipeline already
drops a sub-question identical to the original question before its own fan-out;
without matching that here the lane would search the same text twice through a
different path.

AC #3: the `[SearchSettings] search_enable_subquery` comment now states that
enabling fan-out changes both retrieval and judging, and the eval baseline doc
carries a scope note on every recorded fan-out number -- they measured the
gate-context half only, so fan-out's retrieval value is untested rather than
disproven.

Modified: `tldw_chatbook/Research_Interop/local_research_engine.py`,
`tldw_chatbook/config.py`,
`Tests/Research/test_local_research_engine.py`,
`Docs/Development/research-report-eval-baseline.md`.
<!-- SECTION:NOTES:END -->
