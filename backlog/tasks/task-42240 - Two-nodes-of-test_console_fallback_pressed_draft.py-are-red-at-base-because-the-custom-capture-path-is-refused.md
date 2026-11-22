---
id: TASK-42240
title: "Two nodes of test_console_fallback_pressed_draft.py are red at base because the custom-capture path is refused"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/UI/test_console_fallback_pressed_draft.py
- Tests/UI/test_console_send_acknowledgement.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consolea stream (round 19). `Tests/UI/test_console_fallback_pressed_draft.py::test_custom_capture_keeps_identically_reauthored_pressed_draft[visible]` and `[hidden]` are red on dev e5626ea043. Re-run through the scratch wrapper at `-n 4` (load about 11): both fail at `Tests/UI/test_console_fallback_pressed_draft.py:75` with `AssertionError: timed out waiting for <bound method Event.is_set ...>` raised from `until()` in `Tests/UI/test_console_send_acknowledgement.py:278`, i.e. the event the test waits on is never set. The stream reported two causes: the HeldMcpRead custom-capture path has been refused since 11d71e5db5, and a second one that is not yet named. Neither cause is confirmed here; only the red result and the failure site are.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of each red node is named and fixed (both causes the stream reported, or what the investigation finds instead)
- [ ] #2 Both nodes pass through the scratch wrapper at `-n 4` on a branch carrying the fix
- [ ] #3 The file runs on a PR lane (a whole-file census entry or a lane target), or its absence from every lane is recorded with the reason
<!-- AC:END -->
