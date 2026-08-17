---
id: TASK-17372
title: Sub-question fan-out never reaches the academic lane
status: To Do
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
- [ ] #1 Whether the academic lane searches generated sub-questions is a stated, tested decision rather than a side effect of where sub-question generation lives.
- [ ] #2 If the lane does fan out, its extra searches are counted against the same budget ledger and search cap as the web lane's, with no path that spends past the cap.
- [ ] #3 The gate-versus-retrieval asymmetry is documented wherever the decomposition settings are described, so a measured result cannot be misread as evidence about retrieval.
<!-- AC:END -->
