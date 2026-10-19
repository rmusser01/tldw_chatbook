---
id: TASK-35644
title: Reduce measured Console return preparation work
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-19 21:24'
updated_date: '2026-10-19 21:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Explain the slower Console return comparison and reduce the next demonstrated bottleneck while preserving storage and view authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The investigation separates observed input, styling, native work and scheduling delays and reports limits of causal attribution.
- [x] #2 A focused regression fails before the measured fix and passes afterward without weakening storage safety, freshness or physical retirement.
- [x] #3 Serial native Windows before/after profiles retain all outcomes, exact exits and guard counts.
- [ ] #4 Targeted verification, documentation, review and PR 3166 update are complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: optimize the existing private Textual matcher for one subject compound; no storage, cache, authority, source lifetime or cross-module interface changes. Existing ADR-126/230 display boundaries remain. 1. Preserve native phase/queue discovery and distinguish historical attribution limits. 2. Add a failing subject-only selector regression, then use original Selector.check on the subject when SAME/zero-advance shape proves no ancestry; retain upstream for ancestry and uncertain shapes. 3. Run focused style/state/invalidation and storage/attach checks, then serial original/current ABBA profiles pinned to 48339e4d6d. 4. Independently review, document all results and update PR 3166.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Investigated historical and fresh Console return profiles without changing storage safety. Separate identity-preserving native observations measured 98-135 ms full styling, 388-1332 ms config waits, deferred FULL replay, and a 5.71 s config read crossing leave/return. These identify fresh delay contributors; the historical regression cause remains unproved. Direct subject-only selector matching now avoids ancestor scans when initial DESCENDENT and SAME/zero-advance shape prove equivalence. Original checks, dynamic state, invalidation and ancestry fallback remain. Four review edge controls failed before the guard and pass after it. Final targeted Windows verification:48 passed in135.07s. Two pre-existing native teardown failures reproduced on exact baseline CSS; real-app fixture construction repairs them without changing assertions or retirement gates. Serial ABBA:input medians189.77/170.35/166.20/167.72ms; first pair10.2%better, reverse effectively flat0.9%; attachment16.9%worse/8.8%better. No consistent end-to-end improvement established. CSS median94.32 to91.84ms(2.6%,overlapping), zero style differences513widgets. All24drafts retained, five processes exit0, no network/profile guard events. Eleven safety/ownership modules unchanged; lint/format checks pass; independent review has no remaining findings. ADR required:no; existing boundaries preserved. Evidence in RETURN_MATCHER_RESULTS.md. PR publication pending.
<!-- SECTION:NOTES:END -->
