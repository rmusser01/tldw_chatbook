---
id: TASK-19193
title: Library media-viewer back nav test red on dev (LookupError active_app)
status: To Do
assignee: []
created_date: '2026-08-20'
labels:
  - test-health
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_screen_navigation.py::test_action_library_media_viewer_back_returns_to_list_and_refocuses_it`
is red on dev `7877defba` (re-confirmed 2026-08-20 in a pristine worktree,
isolated config), with the same signature TASK-19046's implementer AND
reviewer both reproduced at their pristine base:

    screen.action_library_media_viewer_back()   # test_screen_navigation.py:2619
    ...
    textual/message_pump.py:257: in app
        return active_app.get()
    LookupError: <ContextVar name='active_app' at 0x...>

The test is synchronous by design: it constructs `LibraryScreen` outside a
running app, stubs `refresh`/`call_after_refresh`/`set_timer`, and pins
task-2856's contract that Escape from the plain read-only media viewer reuses
the exact `_exit_library_media_viewer` reset sequence and re-focuses the
list's first row (one seam for both exits). Something on the
`action_library_media_viewer_back` path now reaches `self.app`, which raises
outside an app context. Diagnose which it is before fixing: a product change
that made the exit path app-dependent (fix or stub at the new seam without
weakening the one-seam contract) versus a test-harness gap (extend the
harness). Per the owner's standing ruling, prefer the durable fix over
whatever silences the test fastest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The test passes on dev without deleting it, skipping it, or weakening the task-2856 one-seam contract it pins (same reset sequence for Escape and the back button, list first-row refocus).
- [ ] #2 Implementation Notes identify the commit/change that introduced the `self.app` access on this path and classify the failure (product regression vs harness gap), with the fix applied at the layer that classification names.
- [ ] #3 The surrounding `Tests/UI/test_screen_navigation.py` suite passes.
<!-- AC:END -->
