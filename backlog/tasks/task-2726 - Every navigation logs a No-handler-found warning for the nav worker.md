---
id: TASK-2726
title: Every navigation logs a "No handler found" warning for the nav worker
status: Done
assignee: []
created_date: '2026-08-06 17:00'
labels:
  - navigation
  - logging
  - uat
  - polish
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During the 2026-08-06 UAT walkthrough on `origin/dev` `b0185749c`, the Logs screen's warning filter was dominated by one line repeated 36 times — once per navigation:

`__main__ WARNING No handler found for worker 'handle_screen_navigation' (Group: screen-navigation)`

`on_worker_state_changed` warns whenever `worker_handler_registry.handle_event` returns unhandled, and the TASK-1230 navigation worker (group `screen-navigation`) has no registered handler, so ordinary tab switching floods the warning channel. With warnings at ~1 per navigation, the Warnings+ filter loses its value as a signal (76 warnings in one browsing session, half of them this line).

Either register a no-op handler for the `screen-navigation` group or exempt known fire-and-forget worker groups from the unhandled-worker warning.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: drive `on_worker_state_changed` with a SUCCESS transition for the `screen-navigation` group under a loguru WARNING sink; guard test proves unknown groups still warn.
2. GREEN: acknowledge known fire-and-forget groups in `MiscWorkerHandler.HANDLED_GROUPS`.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] Tab navigation produces no "No handler found" warnings in the log buffer.
- [x] Genuinely unhandled workers (unknown groups) still produce the warning.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `screen-navigation` to `MiscWorkerHandler.HANDLED_GROUPS`, plus three more fire-and-forget groups the RED run itself exposed producing identical startup noise: `scheduling`, `model-catalog-refresh`, `subscriptions-fts-backfill` (each warned 2-3x in a single test-app boot). Failures remain fully recorded: the diagnostics hook persists `worker_failed` before registry delegation. Guard test pins that genuinely unknown groups still warn. Tests: `test_screen_navigation_worker_transition_does_not_warn_unhandled`, `test_unknown_worker_group_still_warns_unhandled` (Tests/App/test_worker_failure_event.py). Files: tldw_chatbook/Event_Handlers/worker_handlers/misc_worker_handler.py.
<!-- SECTION:NOTES:END -->
