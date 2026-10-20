---
id: TASK-35610
title: Reduce repeated CSS and Console presentation work
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19 17:37'
updated_date: '2026-10-19 18:05'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Windows Console returns spend visible time reselecting unchanged CSS candidates and rebuilding inspector/workspace presentation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CSS results remain identical through class, ancestry, pseudo-state and stylesheet changes while repeated candidate work is reduced
- [x] #2 Inspector and workspace presentation avoids redundant work without reusing storage admission or authority decisions
- [x] #3 Targeted regressions and paired clock-safe Windows profiles document the results and existing PR is updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR-231 and ADR-126 safety boundaries apply. Reason: remove redundant synchronous presentation computation without changing storage, authority, runtime ownership or visual contracts. 1. Measure current Windows returns and trace CSS/inspector callers. 2. Add deterministic regressions for repeated work and invalidation. 3. Implement bounded candidate reuse and narrow presentation deduplication only where measured. 4. Run targeted correctness checks and paired profiles, self/independent review, document evidence and update PR3166.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a bounded candidate-only CSS LRU, removed one unused leading inspector build, and repaired missing/deleted appearance display caching within its existing TTL and service identity. All original style matching and storage checks remain. CSS update median 254 to 195 ms across four interleaved rounds; all 526 base/component styles match. Final paired overall input median worsened 1066 to 1129 ms and worst 1617 to 2523 ms; no end-to-end gain claimed. 31 distinct targeted cases pass across focused runs; two original CSS teardown errors reproduce on baseline. Fixed collection-source fixture selection and separated UI/background SQL counters after original native worker stacks explained a misleading assertion. New code/tests pass Ruff; existing 25 screen/workspace findings unchanged. No new ADR: routine presentation computation changes retain ADR126/231 boundaries. Architecture, evidence and testing lesson updated. PR3166 updated with implementation and measured evidence; evidence at Docs/Development/evidence/windows-performance-2026-10-18/PRESENTATION_RESULTS.md.
<!-- SECTION:NOTES:END -->
