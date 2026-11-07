---
id: TASK-2520
title: Library landing footer advertises the wrong keyboard story
status: Done
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

- [x] The Library landing footer's shortcut hint text matches what the screen actually supports (F6 next pane,
      or whatever the current real binding is) instead of a stale/incorrect hint (e.g. `Ctrl+Q quit`)
- [x] `Tests/UI/test_library_shell.py::test_landing_footer_advertises_the_landing_keyboard_story` passes
- [x] No other footer/keyboard-hint test in the same file regresses

## Closeout audit 2026-10-11

Verdict: DONE (verified against origin/dev @ a3c47c6c72).

Evidence:
- The drift was reconciled by task-3302 (footer appends shared global hints sourced from the footer's own constant so screen copy and globals cannot drift) plus the ctrl+n key change (task-32138). On dev the landing footer advertises a coherent story: `"/ focus search | i import content | ctrl+n new note | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit"` — `F6 next pane` is a real screen binding (`tldw_chatbook/UI/Screens/library_screen.py:1268`).
- The originally-failing test now pins exactly that string: `Tests/UI/test_library_shell.py::test_landing_footer_advertises_the_landing_keyboard_story` (:7010-7038), with the sibling pin in `Tests/UI/test_screen_footer_hints.py` updated in lockstep.
- All three ACs ticked.
