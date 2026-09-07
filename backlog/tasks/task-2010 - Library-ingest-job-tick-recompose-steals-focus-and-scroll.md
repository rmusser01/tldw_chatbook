---
id: TASK-2010
title: >-
  Library ingest job-tick recompose steals focus and scroll
status: Done
assignee: []
created_date: '2026-08-02 21:30'
labels:
  - library
  - ingest
  - ux
  - uat
priority: high
dependencies: []
---

## Description (the why)

Every ingest job transition (`_handle_library_ingest_registry_changed`,
`library_screen.py:5537`) runs a full-screen `refresh(recompose=True)` while
the ingest canvas is selected. A recompose remounts every widget, silently
dropping focus from the Input the user is typing into; later keystrokes hit
the app's global digit bindings and navigate away mid-word (a typed-path
fragment was observed landing in a Console composer). Queue scroll position
also resets on every tick, so watching a folder batch means being yanked to
the top repeatedly. Found in the 2026-08-02 ingest UAT (critique snapshot
`.impeccable/critique/2026-08-02T21-04-04Z__chatbook-widgets-library-library-ingest-canvas-py.md`).

## Acceptance Criteria (the what)

- [x] Typing into the ingest path (or title/author/keywords) field while a
      queued job transitions keeps focus, text, and cursor position.
- [x] Queue scroll position survives a job transition.
- [x] A focused widget that no longer exists after the recompose (e.g. a
      row-action button of a finished job) degrades gracefully: no
      exception, focus falls back to the screen default.

## Implementation Notes

`_handle_library_ingest_registry_changed` now routes through
`_refresh_library_ingest_canvas_preserving_context()`: capture focused
widget id + Input cursor + `LibraryIngestCanvas.scroll_offset.y` before the
recompose, restore via one `call_after_refresh` hop (scroll first, then
`focus(scroll_visible=False)`, cursor clamped to value length). Vanished
ids degrade silently. Same remount-restore family as
`_focus_library_search_input` and the rail's `scroll_to(force=True)`.
Files: `tldw_chatbook/UI/Screens/library_screen.py`,
`Tests/UI/test_library_shell.py` (2 tests: typing-focus preserved through
the listener; vanished-widget restore doesn't raise).
Verified: tests red→green; live (2026-08-02, dev worktree, isolated
profile): typing during a folder batch's transitions no longer teleports
screens, and a mid-canvas scroll position held through the batch settling.
Residual (accepted): keystrokes landing inside the sub-100ms window
between the recompose and the restore callback can still be swallowed —
without navigation side effects now; noted rather than fixed because
eliminating it needs the in-place queue update (larger refactor).
