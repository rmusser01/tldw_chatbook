---
id: TASK-2011
title: >-
  Library ingest stale pre-flight result resurrects a cleared summary
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

`_do_submit_ingest` clears `form.preflight` on purpose (its comment promises
a late result "cannot repopulate what was cleared"), but pre-flight runs as
`@work(thread=True)` whose cancellation is cooperative, and
`_apply_library_ingest_preflight_result` applies unconditionally via
`call_from_thread` (`library_screen.py:12396`). A pre-flight that finishes
after the submit resurrects the cleared summary: observed twice live as
"Enter a file path to start." rendered directly above "1 plain text file ·
277 B" with an empty path field. Found in the 2026-08-02 ingest UAT
(critique snapshot 2026-08-02T21-04-04Z).

## Acceptance Criteria (the what)

- [x] A pre-flight result whose analysis started before a submit/clear never
      repopulates the summary after the clear.
- [x] A pre-flight result for the current (uncleared, unsuperseded) trigger
      still applies normally.

## Implementation Notes

Generation stamp, not better cancellation: `_library_ingest_preflight_generation`
(int, screen-owned) is bumped by a new `_invalidate_library_ingest_preflight()`
helper and by every `_trigger_library_ingest_preflight`; the worker carries the
generation it was started under and `_apply_library_ingest_preflight_result`
drops any result whose generation is no longer current. All three raw-clear
sites now route through the helper: the path Clear button, `_do_submit_ingest`
(whose comment previously promised cancellation was sufficient — it is not,
cancellation is cooperative), and `_reset_library_ingest_transient_state`
(canvas re-entry, which replaces the whole form and was equally exposed).
Files: `tldw_chatbook/UI/Screens/library_screen.py`,
`Tests/UI/test_library_shell.py`
(`test_library_ingest_stale_preflight_result_is_dropped_after_clear`).
Verified: new test red→green; `-k "ingest or preflight"` subset 29/29.
