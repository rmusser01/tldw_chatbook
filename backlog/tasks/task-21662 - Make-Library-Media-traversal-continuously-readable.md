---
id: TASK-21662
title: Make Library Media traversal continuously readable
status: Done
assignee: []
created_date: '2026-08-24 06:57'
updated_date: '2026-08-24 14:47'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Library Media row traversal, Reader detail loading, and authoritative filtering continuous and truthful across settlement delays, stale responses, pagination, and Select mode.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Arrow traversal selects immediately, settles detail after 120 ms, and Enter loads immediately.
- [x] #2 Canonical media id plus session generation reject stale success and failure without clearing the last loaded detail.
- [x] #3 Items rows and Reader text truthfully distinguish selected/loading/loaded state, including retryable detail failures.
- [x] #4 Filter media uses authoritative paginated search, restores the prior unfiltered page/id, and reaches records above ordinal 50 without duplicates.
- [x] #5 Select mode cancels pending settlement while preserving the existing loaded Reader and bulk/trash behaviors.
- [x] #6 Focused tests, inverse mutations, static checks, and self-review pass with no regressions in the scoped Media flows.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add event-controlled RED tests for selection settlement, immediate Enter, stale generations, truthful text, authoritative filtering, anchor restoration, and Select cancellation.
2. Implement the smallest session-state, timer, row rendering, and screen orchestration changes that make each test GREEN while retaining the existing 20-row Previous/Next controller.
3. Run the focused Task 3 suite and required inverse mutations, then complete scoped static checks and self-review.

ADR required: yes
ADR path: backlog/decisions/084-library-media-reader-ia.md
Reason: This task directly implements the accepted identity, loading, pagination, and permanent Reader boundaries in ADR-084; no new architectural decision is introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented immediate Items selection with a 120 ms traversal settlement timer, immediate Enter activation, and canonical-id plus generation fencing at every async commit point. Reader retains the last loaded item while a new selection is pending, exposes textual Loading/Loaded row states, and provides a Reader-local Retry action after detail failures.

Added authoritative Filter media requests through the existing paginated controller, including first-result Reader selection and restoration of the captured unfiltered page/id anchor. Entering Select mode invalidates pending single-item work without changing established bulk, receipt, pagination, or Trash behavior.

Verification: the five-file focused Media suite passed 145 tests; required inverse mutations each failed their sentinel before restoration; Ruff, compileall, and `git diff --check` passed. A bounded external review timed out, so direct self-review identified and fixed missing Reader-session advancement on filter application and missing Reader-local retry affordance, both with red-to-green integration coverage.

ADR required: yes
ADR path: backlog/decisions/084-library-media-reader-ia.md
Reason: The implementation follows ADR-084's existing Reader identity, loading, and authoritative pagination boundaries; no new ADR was needed.
<!-- SECTION:NOTES:END -->
