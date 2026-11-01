---
id: TASK-35644
title: Reduce measured Console return preparation work
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-19 21:24'
updated_date: '2026-10-19 22:00'
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
- [x] #4 Targeted verification, documentation, review and PR 3166 update are complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: optimize the existing private Textual matcher for one subject compound; no storage, cache, authority, source lifetime or cross-module interface changes. Existing ADR-126/230 display boundaries remain. 1. Preserve native phase/queue discovery and distinguish historical attribution limits. 2. Add a failing subject-only selector regression, then use original Selector.check on the subject when SAME/zero-advance shape proves no ancestry; retain upstream for ancestry and uncertain shapes. 3. Run focused style/state/invalidation and storage/attach checks, then serial original/current ABBA profiles pinned to 48339e4d6d. 4. Independently review, document all results and update PR 3166.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented direct subject-only CSS matching within the existing private matcher, retaining original selector checks, live state, invalidation and ancestry fallback. Independent review found an initial-combinator edge; four controls reproduced it before the guard and passed afterward. No findings remain. Eleven safety/ownership modules and seven reminder-fencing methods are unchanged. ADR required: no; existing ADR-126/230 boundaries remain.

Separate native observations identify full-screen styling, config/workspace waits, deferred FULL replay and background reads crossing return boundaries. The exact historical regression cause remains unproved. CSS median was 94.32 to 91.84 ms (2.6%, overlapping timings), with zero style differences across 513 widgets. Matcher-only ABBA was mixed. The final targeted native batch passed 48 checks; two baseline-reproduced mocked-app teardown failures were repaired using real app construction without changing assertions or retirement gates.

Preserved concurrent TASK-35615's redundant acknowledgement FULL correction by rebasing onto efdd8b2b21. Integration added no further product changes. Twenty-six overlapping combined rechecks passed. Fresh combined ABBA retained all 24 drafts and attachments, with four zero exits and no guard events. Pooled input was essentially flat (170.57/170.14 ms); attachment worsened (1127.89/1379.36 ms), with a worse 3.58-second tail. No consistent end-to-end or tail improvement is claimed. All adverse results remain in RETURN_MATCHER_RESULTS.md and exact receipts. Lint/format checks pass; no full suite. Twenty exact disposable roots were removed; the cleanup receipt retains its validation parameter error and corrected post-check. Published code and all results to PR #3166, including its leading adverse-result summary. Final independent review recommends retaining auto-merge off: the overall performance objective remains unqualified. Keep this task In Progress until that qualification is resolved.
<!-- SECTION:NOTES:END -->
