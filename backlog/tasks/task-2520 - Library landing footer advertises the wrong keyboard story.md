---
id: TASK-2520
title: Library landing footer advertises the wrong keyboard story
status: To Do
assignee: []
created_date: '2026-08-06 02:16'
labels:
  - library
  - footer
  - bug
dependencies: []
priority: low
---

## Description

`Tests/UI/test_library_shell.py::test_landing_footer_advertises_the_landing_keyboard_story` fails on `dev` at
branch base `afebcad5f` — controller-verified in a throwaway worktree checked out at that exact commit, then
removed. The Library landing footer's shortcut hint text has drifted: it now shows `… · Ctrl+Q quit` where the
test pins `… F6 next pane`. This is the same class of defect as the previously-filed "RAG-36 footer lies about
the keyboard story" finding — the footer's advertised shortcuts don't match what the screen actually does.

This is pre-existing on `dev`, not introduced by the `feat/rag-truth-paid-moments` branch (PR-T2). It was found
only because PR-T2 Task 7's fix round ran the full `test_library_shell.py` file and had to distinguish its own
regressions from unrelated pre-existing failures.

## Acceptance Criteria

- [ ] The Library landing footer's shortcut hint text matches what the screen actually supports (F6 next pane,
      or whatever the current real binding is) instead of a stale/incorrect hint (e.g. `Ctrl+Q quit`)
- [ ] `Tests/UI/test_library_shell.py::test_landing_footer_advertises_the_landing_keyboard_story` passes
- [ ] No other footer/keyboard-hint test in the same file regresses
