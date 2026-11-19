---
id: TASK-35611
title: Reduce Console return CSS matching work
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19 18:29'
updated_date: '2026-10-19 18:50'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate the observed Windows Console return regression and reduce measured remaining CSS matching work while preserving styling and native storage safety.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repeated native Windows profiles distinguish reproducible return costs from variation and record all outcomes
- [x] #2 Measured CSS matching work is reduced with identical resolved styles under dynamic classes ancestry pseudo-state and source changes
- [x] #3 Targeted checks pass or baseline failures are documented and PR3166 contains the verified changes and evidence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR126/231 apply. Reason: conservative presentation optimization preserves CSS semantics, storage policy and runtime boundaries. 1. Repeat current native return profiling and observe remaining selector checks separately. 2. Pin measured redundant matching with failing tests and implement a conservative candidate-filter improvement. 3. Verify real screen style fidelity and dynamic invalidation, then run interleaved CSS timing and paired native return profiles. 4. Review changes, record honest timing limits, and update PR3166.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Strengthened candidate-only filtering to reject impossible static ancestor/subject type, class and ID requirements across comma alternatives. Upstream CSS matching and storage safety remain intact. 12 targeted tests pass including real Console/Library/Settings styles; lint/format pass. Matching calls 16960 to 4233 with 2059 matches and zero style differences on 526 widgets. Warm CSS median 205 to 131 ms. Completed native return pairs improve median 972 to 858 ms and 911 to 815 ms; all drafts retained and attaches complete. One earlier diagnostic exited abnormally with a partial native dump, is excluded and retained, and did not recur in four subsequent runs. No universal latency or crash-fix claim. Independent review found no blockers. ADR not required: existing ADR126/231 boundaries unchanged. Full evidence and reproduction: Docs/Development/evidence/windows-performance-2026-10-18/MATCHING_RESULTS.md. PR3166 updated with verified changes and complete results.
<!-- SECTION:NOTES:END -->
