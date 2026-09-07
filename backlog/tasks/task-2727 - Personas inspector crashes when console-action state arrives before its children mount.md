---
id: TASK-2727
title: >-
  Personas inspector crashes when console-action state arrives before its
  children mount
status: Done
assignee: []
created_date: '2026-08-06 17:20'
labels:
  - roleplay
  - bug
  - race
  - uat
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Surfaced by the live verification pass for the 2720-2726 batch: after a first-run walkthrough, one Roleplay visit logged two chained tracebacks — `PersonasScreen._load_after_mount` → `_sync_inspector_console_actions` → `PersonasInspectorPane._apply_action_state` → `NoMatches: '#personas-validation-summary'`.

`_load_after_mount` runs as a worker; when its awaited loads complete before the inspector pane's composed children finish mounting, the five unguarded `query_one` calls in `_apply_action_state` raise. The race is timing-dependent (not reproduced on a pristine `75bc25db3` in one trial; fired on the branch build in one trial — the batch's footer-timer fix changes first-run timer activity, which perturbs the mount window).

The damage is worse than log noise: the exception aborts `_load_after_mount` midway, so `_apply_pending_restore` and `_auto_select_first_library_row` are silently skipped — a lost screen restore and no initial selection. Same defect family as task-2721 (timer/worker callback assuming mounted children).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: call `set_console_actions_enabled` on an unmounted pane (must not raise), then mount and assert the deferred state is replayed onto the readiness line.
2. GREEN: `_apply_action_state` defers quietly on `QueryError`; `on_mount` replays via `call_after_refresh` so no pushed state is dropped.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] Pushing console-action state to a not-yet-mounted inspector neither raises nor gets lost — the state is applied once the children mount.
- [x] `PersonasScreen._load_after_mount` can no longer be aborted by this race (the inspector call is exception-free pre-mount).
- [x] A regression test drives the pre-mount push and the post-mount replay.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two-part fix in `PersonasInspectorPane`: `_apply_action_state` probes its first child query under `QueryError` and defers quietly when the composed children don't exist yet (the pre-mount push window `PersonasScreen._load_after_mount`'s worker can hit), and a new `on_mount` replays the retained state via `call_after_refresh` so an early push is applied rather than dropped — replay reads live attributes, so it cannot clobber newer state. The regression test drives the exact crash observed live (`set_console_actions_enabled` on an unmounted pane with a selection state) and asserts the pushed reason lands on the readiness line after mount; watched RED with the verbatim `NoMatches '#personas-validation-summary'` from the incident. Scope note: `show_selection`/`clear_selection` also query children unguarded but are not reachable in the observed race window; left alone deliberately. Files: tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py, Tests/UI/test_personas_inspector_pane.py.
<!-- SECTION:NOTES:END -->
