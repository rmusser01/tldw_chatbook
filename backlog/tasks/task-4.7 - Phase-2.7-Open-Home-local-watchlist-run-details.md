---
id: TASK-4.7
title: 'Phase 2.7: Open Home local watchlist run details'
status: Done
assignee: []
created_date: '2026-05-03 20:25'
updated_date: '2026-05-03 20:30'
labels:
  - unified-shell
  - phase-2
  - home
  - watchlists
  - details
dependencies: []
parent_task_id: TASK-4
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Home local W+C watchlist run detail controls open the existing W+C runs surface with the relevant run selected and loaded, instead of showing an unavailable active-work adapter message.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home adapter handles `Open details` for visible local watchlist run active-work items and keeps unknown/non-run targets recoverable.
- [x] #2 Home detail navigation stages the W+C runs tab and selected run id before opening subscriptions.
- [x] #3 SubscriptionWindow consumes the pending Home-selected watchlist run id, restores the run selection, and loads its detail when local runs are available.
- [x] #4 Local W+C runs remain visible in the runs tab while unrelated local-only server controls stay honest.
- [x] #5 Focused automated tests and Phase 2 QA evidence verify the adapter, app navigation, W+C run detail workflow, and tracking updates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for Home adapter local watchlist detail handling, app detail navigation context, SubscriptionWindow pending run consumption, and local W+C runs visibility/detail loading.
2. Implement the smallest adapter, app, and SubscriptionWindow changes needed to route local watchlist run details without enabling unrelated controls.
3. Add Phase 2 QA evidence and update roadmap/task tracking.
4. Run focused Home, W+C, and tracking tests plus diff hygiene before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added handled Home detail routing for visible local W+C watchlist run items, including app-level staging of the W+C runs tab and selected run id before navigation. `SubscriptionWindow` now consumes that pending run context, restores the run selection, and loads run detail JSON while local W+C runs and alert rules remain visible through the existing scope service. Local watchlist jobs stay explicitly recoverable as server-style control-plane functionality. Added focused adapter, Home hook, SubscriptionWindow workflow, and Phase 2 tracking coverage plus QA evidence for `TASK-4.7`.
<!-- SECTION:NOTES:END -->
