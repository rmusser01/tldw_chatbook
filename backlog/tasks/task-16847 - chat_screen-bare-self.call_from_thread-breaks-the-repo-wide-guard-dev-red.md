---
id: TASK-16847
title: 'chat_screen bare self.call_from_thread breaks the repo-wide guard (dev red)'
status: Done
assignee:
  - '@claude'
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
- [x] #1 `Tests/test_call_from_thread_guard.py` passes on dev (both tests)
- [x] #2 The corrected call path is exercised once for real (or by test) so the presented snapshot is shown to actually work, not just to stop raising
- [x] #3 The guard is not weakened (no new allowlist entry for chat_screen.py)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline at `ecbcd5cd8`: run the guard file (expect the 3054 red) and
   `Tests/UI/test_trajectory_live.py` (expect green — established the masking).
2. Read the site in context: `action_open_trajectory_view` — a thread worker
   builds the snapshot, then marshals `present` back. Verify against installed
   Textual 8.2.8 what `Screen` actually has (probe found: NEITHER
   `call_from_thread` NOR `push_screen` — `present` is broken twice over).
3. `git log -S` for the introducing commit; check the same diff for other bare
   sites and for allowlist absorption.
4. Fix: `self.app.call_from_thread(present, snapshot)` at the worker seam and
   `self.app.push_screen(...)` inside `present` (the AC#2 "actually work" check —
   fixing only the marshal would move the AttributeError one frame later).
5. Rewrite `test_trajectory_launch_action_presents_screen`: the shipped version
   monkeypatched `instance.call_from_thread`/`instance.push_screen` — doubles for
   attributes that do not exist on Screen, which is exactly how the bug shipped
   green. Replace with a real-app async test: graft the action onto a minimal
   running App so the real thread worker, real `App.call_from_thread`, and real
   `App.push_screen` execute, and assert a `TrajectoryScreen` lands on the stack.
6. Re-run the guard file (both tests) + the trajectory file; ruff on touched
   files; document why dev being red raised no alarm.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**The bug was twice as deep as filed.** The site (`action_open_trajectory_view`,
the Console `y` binding — thread worker builds a `TrajectorySnapshot` off the UI
thread, marshals `present` back) was broken at BOTH seams: `Screen` in Textual
8.2.8 defines neither `call_from_thread` nor `push_screen` (probed:
`hasattr(Screen, ...)` is False for both — App-only). Fixing only the filed
`self.call_from_thread(present, snapshot)` would have moved the AttributeError
one frame later, into `present`'s bare `self.push_screen(...)`. Both are now
`self.app.` (`UI/Screens/chat_screen.py` ~3042/3058, with a task-16847 comment).
No broad except swallows this one, contrary to the filing's guess: the worker
runs with `exit_on_error=True` defaults, so pressing `y` on a persisted Console
conversation would have errored the worker, not just silently dropped the screen.

**Introducing commit:** `a8082fe85` (2026-08-14, "feat(console): trajectory
screen launch + live tail-follow", via the task-15791 console-clusters branch).
No allowlist absorption — the same diff's other new call is the CORRECT
`self.app.call_from_thread` in `trajectory_screen.py`, so the bare form was a
one-off slip, not a pattern.

**Why green tests shipped it:** the same commit's own launch test
(`Tests/UI/test_trajectory_live.py::test_trajectory_launch_action_presents_screen`)
monkeypatched `instance.call_from_thread` and `instance.push_screen` directly
onto the `ChatScreen.__new__` instance — doubles for attributes that do not
exist on the class at all. Rewrote it as a real-seam async test: a minimal real
App under `run_test()`, `Screen.__init__` + `_parent` graft so `self.app`
resolves, then the REAL `run_worker(thread=True)`, REAL `App.call_from_thread`
marshal, and REAL `App.push_screen`; asserts a `TrajectoryScreen` lands on the
app's screen stack. Mutation-checked: reverting either seam to the bare form
fails the new test (both mutants killed, then restored via Edit).

**Why the red guard raised no alarm on dev:** CI checks are intentionally
cancelled in this repo (local verification is the gate), and the programme's
targeted-tests rule means a chat_screen.py implementer runs `Tests/UI/...`
suites — the repo-wide guard lives at `Tests/test_call_from_thread_guard.py`
(top level) and is only exercised by a full-suite run. It argues for the guard
file being added to whatever fast gate the merge flow actually runs; left as a
programme-level observation rather than a config change here.

**Evidence:** baseline at `ecbcd5cd8`: guard 1 failed (chat_screen.py:3054),
trajectory file 9 passed (masked). After fix: guard 2 passed + trajectory 9
passed (11 total). Ruff clean on touched files (the one F401 `inspect` hit in
chat_screen.py pre-exists at base, untouched). Swept the package for the same
push_screen family: app.py sites are on the App, `Console_Modules/*` controllers
define their own delegating `push_screen` (documented pattern), Third_Party hits
are vendored example Apps — chat_screen.py:3042 was the only real one.

**Files:** `tldw_chatbook/UI/Screens/chat_screen.py` (2-line fix + comment),
`Tests/UI/test_trajectory_live.py` (real-seam rewrite of the launch test),
`backlog/docs/lessons-testing-evidence.md` (addendum to "A fake written to
match your call site": an instance monkeypatch is also an existence claim).
<!-- SECTION:NOTES:END -->
