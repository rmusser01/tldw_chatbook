---
id: TASK-34931
title: Move-Schedules-results-overlay-and-mark-read-queries-off-the-UI-thread
status: Done
assignee: []
created_date: 2026-10-13 14:54
labels:
- performance
- ui-perf
- wave-4-blocking-rebuilds
dependencies: []
priority: medium
updated_date: 2026-10-13 18:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clicking the Schedules Results badge runs list_automation_results(limit=200) + count_automation_results + full list_automation_definitions inline on the UI thread before push_screen (schedules_workbench.py:2483); the r mark-read binding reads the unread set inline (:3808) and _push_conflicts_overlay reads get_conflicts inline (:2541). _read_counts in the same file already uses asyncio.to_thread, so this is an inconsistency, not a convention — click/keypress lag scales with backlog size.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Overlay and mark-read query prep runs via to_thread or a worker with handlers only orchestrating
- [x] UI stays responsive with a large results backlog (targeted test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read _push_results_overlay/_read_counts + mark-read paths
2. Move the sync query prep to asyncio.to_thread like _read_counts
3. Tests + targeted run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `_push_results_overlay` and `_push_conflicts_overlay` are now async (their Button.Pressed/event callers too); the three-read inbox query and the `get_conflicts` read run via `asyncio.to_thread`, matching `_read_counts`'s existing pattern. `action_mark_read` (`r`) and the rail's mark-all move their unread reads (count + list) INSIDE their dispatch workers, also via to_thread — the "Nothing unread" notifications now fire from the worker, which runs on the loop so `notify` is unchanged.
- Tests: new `Tests/UI/test_schedules_offloop_reads.py` (2 tests asserting the overlay/conflicts reads execute on a non-event-loop thread and the push still happens). 2/2 pass.
- Regression: `Tests/UI/test_schedules_workbench.py` + `test_schedules_automations_tab.py` are at their exact pre-existing baseline (227 errors with the change stashed — same mount breakage family).
- ADR required: no — threading change matching an established pattern in the same file.
