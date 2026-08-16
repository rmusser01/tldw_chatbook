---
id: TASK-16847
title: 'chat_screen bare self.call_from_thread breaks the repo-wide guard (dev red)'
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - bug
  - test-health
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/test_call_from_thread_guard.py::test_no_bare_self_call_from_thread_outside_app`
(the TASK-929 repo-wide backstop) is red on dev — re-verified at `ee741cf10` this
session:

```
AssertionError: found bare 'self.call_from_thread(' outside App subclasses ...
    UI/Screens/chat_screen.py: lines [3054]
```

The site is `self.call_from_thread(present, snapshot)` at
`UI/Screens/chat_screen.py:3054` (line 3029 when the 15991 review first caught it, PR
#1701 — the file has since shifted; the review also confirmed the identical red at the
then-current origin/dev tip, so this pre-dates and post-dates that whole burn-down).
`ChatScreen` is a `Screen`, not an `App` — only `App` defines `call_from_thread` — so
when this thread-worker path executes, it raises `AttributeError` instead of presenting
the snapshot, most likely swallowed by a broad except (the exact TASK-929 failure mode:
the notification path breaks precisely when it is needed, invisibly).

Fix is the standard `self.app.call_from_thread(present, snapshot)`, plus a check of what
`present`/`snapshot` actually surface so the newly-working call does not present broken
copy — and find why the guard being red on dev raised no alarms (it argues for the guard
running in whatever gate the merge queue actually exercises).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `Tests/test_call_from_thread_guard.py` passes on dev (both tests)
- [ ] #2 The corrected call path is exercised once for real (or by test) so the presented snapshot is shown to actually work, not just to stop raising
- [ ] #3 The guard is not weakened (no new allowlist entry for chat_screen.py)
<!-- AC:END -->
