---
id: TASK-14920
title: >-
  Repair 20 stale-contract failures in the console and personas UI suites
status: To Do
assignee: []
created_date: '2026-08-11 02:00'
labels:
  - tests
  - console
  - personas
  - dev-baseline
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by task-14912's sweep, which ran every affected `Tests/UI` file WHOLE for the first time (its AC#4 exists because a file that has ever contained a hang has an unknown pass count). Two files carry failures nobody had counted:

- `Tests/UI/test_console_native_chat_flow.py` — **291 passed / 18 failed**
- `Tests/UI/test_personas_workbench.py` — **310 passed / 2 failed**

**These are NOT caused by the bounding work and were NOT hidden by a hang.** Every one was reproduced against the pristine `eb9708cc4` copy of each file (`git show HEAD:<path>` into a temp file, run, delete), producing identical failure sets. They are **stale-contract breakage from the screen-decomposition programme**: the tests still call seams that moved.

The dominant shape (14 of the 18) is `AttributeError: 'ChatScreen' object has no attribute '_ensure_active_console_session_settings'` — that seam moved onto `ChatScreen._session` when the session controller was extracted. Others are behavioural: `assert store.identity_at_append is not None`, `assert [] == ['Hello User, I am Elara.']`.

This matters beyond the count. The screen-decomposition programme's own lesson is that *extraction cannot outrun growth* and that a one-way ratchet is what makes a gain stick — but a ratchet measures size, not whether the suites that pin the extracted behaviour still run. Twenty failing tests in the console's main flow suite are twenty behaviours nobody is actually checking, and the longer they sit the more they read as background noise (the "a suite that no gate runs can rot invisibly" lesson, one directory over).

Each failure needs triage before repair: a moved seam is a test fix, but `identity_at_append is None` and an empty greeting list may be real product regressions. Do not mass-rewrite to green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Each of the 20 failures is classified as stale-contract (test fix) or real regression (product fix), with the evidence that decided it — not repaired wholesale to green
- [ ] #2 Any classified as a real regression is fixed in the product, or filed separately with its reproduction if it needs owner judgement
- [ ] #3 Both files run WHOLE with a READ nonzero pass count and zero failures
- [ ] #4 If the moved-seam shape (`_ensure_active_console_session_settings` and friends) recurs across other suites, the sweep that finds them is checkable rather than asserted
<!-- AC:END -->
