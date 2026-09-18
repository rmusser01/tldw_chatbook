---
id: TASK-32702
title: Restore saved Console sidebar state without startup errors
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 20:33'
updated_date: '2026-09-16 20:41'
labels:
  - ui
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The native Import restart qualification exposed a caught Console sidebar-state load failure. Preserve saved sidebar preferences across startup and keep the existing deferred persistence contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh Console construction restores saved sidebar expansion, search and section state without a caught startup error.
- [x] #2 Subsequent sidebar changes retain debounced persistence and immediate-quit flushing.
- [x] #3 Targeted regressions and isolated native restarts verify restored state and clean sidebar startup diagnostics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: restore the existing sidebar persistence contract by correcting constructor ordering; no storage format, authority, styling or lifecycle boundary changes.

1. Add a real ChatScreen construction and mounted restoration regression using a saved private ui_state.toml; confirm the missing timer error on the unchanged source.
2. Initialize existing debounce, dirty and worker fields before loading the reactive sidebar state. Preserve existing debounce and quit flushing.
3. Run sidebar restoration, persistence and relevant lifecycle tests, scoped static checks, and an isolated native save/restart check with log and exit receipts.
4. Review the bounded diff, document evidence and update task status.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ChatScreen now initializes its existing sidebar debounce timer, dirty flag and persistence worker before loading saved reactive state. This prevents the first watcher call from raising AttributeError and avoids resetting persistence state established during restoration. Storage, debounce, quit flushing and UI styling contracts are unchanged; no new ADR required.

Added real-constructor regressions for empty and populated saved sidebar mappings, including search/section restoration and a new toggle persisted on immediate teardown. Both cases failed on baseline df121a4169. Final sidebar/path selection: 10 passed in 15.90s. Two adjacent suspend tests fail identically on baseline/current because their fake fleet lacks _console_wake_user_priority; those unrelated failures remain explicit. No full suite ran.

Two isolated native app processes restored the seeded values and a change made before Quit, returned normally, exited 0 and were independently confirmed absent. The private app log has no ERROR/CRITICAL or traceback entries; ten private databases pass integrity checks, messages/conversations remain empty and default-profile hashes are unchanged. One final Console capture inspected; this verifies stored sidebar state rather than visible rail gestures. Evidence: Docs/superpowers/qa/2026-09-16-sidebar-startup/README.md.

The test and native helper pass Ruff lint/format; screen fatal checks and edited-range formatting pass, with the same 212 inherited Ruff diagnostics and unrelated whole-file formatting debt. Independent read-only review found no actionable issues. Updated lessons-testing-evidence.md with the saved-profile constructor gap. No provider calls, unrelated process cleanup, schema or authority changes.
<!-- SECTION:NOTES:END -->
