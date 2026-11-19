---
id: TASK-39402
title: Qualify frozen dev cut for 0.2.4 release to main
status: In Progress
created_date: 2026-11-08 22:07
assignee:
- '@codex'
labels:
- release
- testing
priority: high
updated_date: 2026-11-08 22:07
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare release0.2.4 from frozen remote-dev commit a0ae30b7ed9c313fe19cb3a0da5d95ebde182385, preserving the previous candidate and selecting only repairs necessary on this cut. The initial full local condition is established; failures remain release blockers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Frozen cut and preserved prior worktree are recorded; later dev changes are excluded except explicitly selected critical fixes.
- [ ] #2 One consolidated initial result covers all5670 first-party test files, with failed collectors, process/accounting gaps, skips and expected failures explicit.
- [ ] #3 One blocker list records root cause, necessary fix and actual verification for every failing test or release requirement; unrelated redesign is excluded.
- [ ] #4 Approved0.2.4 decisions, Lightning deferral, tokenizer security guard and existing architecture limits are preserved.
- [ ] #5 Final candidate passes full integrated local tests and packaging/fresh-install checks; any unavailable requirement or remaining failure is explicitly reported.
- [ ] #6 Verified changes are committed/pushed to draft PR3475 to main; readiness is claimed only after all release blockers are cleared.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR for release coordination.
ADR path: N/A; atomic repairs record applicable existing ADRs or create one when required.
Reason: freezes and qualifies an existing release without a new architectural contract.
1. Preserve the old candidate and freeze a0ae30b7ed9c313fe19cb3a0da5d95ebde182385 (done).
2. Consolidate386 primary and4 supplemental initial attempts with exact unsuccessful identity notes and process/parser limitations.
3. Repair reproduced root causes in atomic tasks, using existing approved source reviews and focused checks; keep one blocker list.
4. Save and push verified units to draft PR3475; exclude later dev wholesale.
5. Run final resource-safe full integrated tests, required local prerequisites,0.2.4 app-only packaging/native CPU wheel/fresh installs and final review.
6. Update the PR to main with final evidence and explicit blockers; do not merge, tag or publish.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
