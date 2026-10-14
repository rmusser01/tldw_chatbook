---
id: TASK-34937
title: Stop-spawning-webbrowser-open-and-pyperclip-on-the-UI-thread
status: Done
assignee: []
created_date: 2026-10-13 14:57
labels:
- performance
- ui-perf
- wave-4-blocking-rebuilds
dependencies: []
priority: low
updated_date: 2026-10-13 19:20
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Link handlers spawn the OS open command on the UI thread (settings_screen.py:23600, console_transcript.py:1964, library_media_controller.py:4204, library_collections_controller.py:1583) and the inspector copy spawns pbcopy (console_conversation_inspector.py:1910). Each spawn can cost 50-150ms of UI freeze.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Link and copy actions dispatch subprocess spawns via to_thread or executor
- [x] No subprocess spawn on the event loop in the named handlers
<!-- AC:END -->


## Implementation Notes

- All five spawn sites (settings link handler, console transcript `_open_link`, library media + collections URL openers, inspector `pyperclip.copy`) now dispatch through `run_worker(..., thread=True, exit_on_error=False)` — the OS spawn (50-150ms freezes) never runs on the loop, and a spawn failure can no longer kill the app.
- Behavior note: the inspector's success toast now fires when the copy is DISPATCHED (worker failures are swallowed by exit_on_error rather than caught inline); the collections opener's except branch remains for the pre-spawn state reads.
- Verification: all five edited modules import clean; `test_console_raw_cli_transcript.py` (exercises the transcript link-handler file) 17/17 green.
- ADR required: no — spawn dispatch change at five call sites.