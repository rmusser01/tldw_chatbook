---
id: TASK-157
title: Chatbook export progress reporting and cancel
status: Done
assignee: []
created_date: '2026-07-11 22:01'
labels:
  - follow-up
  - export
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Library export worker shows a static Exporting… line because ChatbookCreator has no progress callback or cancel hook. Add progress hooks to the creator and surface a progress line + cancel control in the export form. v1 deliberately shipped without this.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Export form shows real progress while a large export runs
- [x] #2 User can cancel an in-flight export
- [x] #3 Creator exposes progress/cancel hooks
<!-- AC:END -->

## Implementation Notes

Implemented across five commits (T1-T5):

- **T1**: `ChatbookCreator` (`tldw_chatbook/Chatbooks/chatbook_creator.py`)
  grew `progress_callback`/`cancel_event` hooks, emitting phase/current/
  total ticks during selection/media/conversation/notes collection and
  checking the cancel event between phases; finalize (zip write + manifest)
  is atomic so a cancel never leaves a partial archive.
- **T2**: `export_chatbook` (`tldw_chatbook/Chatbooks/local_chatbook_service.py`)
  forwards `progress_callback`/`cancel_check` to the creator and returns a
  `cancelled: bool` alongside `success`/`message`/`path`/`dependency_info`.
- **T3**: `tldw_chatbook/Library/export_progress.py` (new) — a pure
  `ExportProgressThrottle` (rate-limits UI ticks) and
  `format_export_progress_line` (phase/current/total -> display string),
  unit-tested independent of Textual.
- **T4**: The Library export worker (`tldw_chatbook/UI/Screens/library_screen.py`)
  wires the throttle + formatter into a `_apply_library_export_progress`
  targeted status-line update (no recompose), and a per-run
  `threading.Event` is created at submit and threaded through as
  `cancel_event` (not yet consumed by anything until T5).
- **T5**: A `#library-export-cancel` Button in
  `tldw_chatbook/Widgets/Library/library_export_canvas.py` (display-toggled
  by `state.running`, same pattern as the canvas's other always-mounted/
  toggled widgets) sets that Event via `handle_library_export_cancel`
  without bumping `_library_export_run_id` (the run is still current until
  the worker reports back). The worker's outcome dict now carries
  `cancelled`; `_run_library_export_worker` branches cancelled ->
  `_marshal_library_export_cancelled` before the success/failure branches.
  `_apply_library_export_cancelled(run_id)` mirrors the failure path's
  staleness guard (`run_id != self._library_export_run_id` bails before any
  mutation), then sets `_library_export_status = "Export cancelled."` and
  calls `_update_library_export_canvas_after_run()`, which now also hides
  the Cancel button. `_reset_library_export_transient_state`
  (navigate-away) sets the outgoing Event before bumping `run_id` so an
  abandoned worker stops promptly.

Tests: `Tests/Chatbooks/test_chatbook_creator.py`,
`Tests/Chatbooks/test_local_chatbook_service_export.py`,
`Tests/Library/test_export_progress.py`,
`Tests/UI/test_library_export_progress_apply.py`,
`Tests/UI/test_library_export_cancel.py` (new).
