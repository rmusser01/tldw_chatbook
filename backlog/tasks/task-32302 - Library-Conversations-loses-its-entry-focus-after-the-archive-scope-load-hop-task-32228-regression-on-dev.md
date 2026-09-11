---
id: TASK-32302
title: >-
  Library Conversations loses its entry focus after the archive-scope load hop
  (task-32228 regression on dev)
status: To Do
assignee: []
created_date: '2026-09-11 00:40'
labels:
  - library
  - focus
  - conversations
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32228 gave Conversations the entry focus every other browse list has, which is what makes its Escape hop live and its footer chip honest. Measured green 24/24 on fix/library-crit9-shell before merging dev adb7d5886f; measured 4 of 5 runs RED immediately after, with no change to the branch's own code. The rows are not late and the 2s arm window is not the constraint: the rows mount 0.44s after the rail-row press with 1.75s of window still open, the arm is still pending, and nothing re-requests focus. The arm generation reaches 2 on every run, so a second arm fires during the route entry and invalidates the first scheduled attempt. dev's change in scope is the archive-scope recovery annotate hop added to the conversations page load (library_conversation_recovery.py, library_conversations_controller.py). Separate from task-32301, which is about the fixed 2s window on genuinely slow loads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening Conversations from the rail lands focus on its first row on every run, not one in five
- [ ] #2 The 'esc focus Library' footer chip is present on arrival, as task-32228 delivered
- [ ] #3 Tests/UI/test_library_crit9_shell.py::test_the_conversations_footer_advertises_escape_on_arrival passes without xfail
<!-- AC:END -->
