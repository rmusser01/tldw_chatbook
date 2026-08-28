---
id: TASK-23152
title: >-
  Library prompts canvas sync swallows its exception so its failures cannot be
  diagnosed
status: To Do
assignee: []
created_date: '2026-08-28'
labels:
  - library
  - diagnostics
priority: high
dependencies: []
---

## Description

`Tests/UI/test_library_prompts_canvas.py` has 3 failing tests when run with other UI files and
**5** when run alone (the file is order-sensitive in both directions). Symptoms are a 30-second
"Prompt browse never settled" wait, a `TextArea` identity change indicating the content pane was
remounted when the test expects it preserved, and `NoMatches` on the conflict bar. Runs are
littered with `Library prompts canvas sync failed.`

**A verdict cannot be reached in the current code.** That log line is a bare `except Exception`
that discards the error, unlike every sibling handler in the same file, so the real exception is
invisible. Restoring the traceback is a prerequisite for diagnosis, and is worth doing on its own
merits.

## Acceptance Criteria

- [ ] The canvas-sync failure path logs its exception (`logger.opt(exception=True)`), matching the
  sibling handlers in the same module
- [ ] With the traceback available, each failing test is classified stale-vs-broken and fixed
  accordingly, with the actual error recorded in the implementation notes
- [ ] The file passes both standalone and in a whole-directory run (its order-sensitivity is
  resolved, not worked around by a run-order constraint)

## Evidence

`tldw_chatbook/UI/Screens/library_screen.py:1939` — `except Exception: logger.debug(f"Library
{kind} canvas sync failed.")`, with no `opt(exception=True)`; siblings at `:1896`, `:1915`, `:1944`
do carry it. Failing tests date from `eaaddb1f5e` (2026-07-12), so the break is recent; the
plausible cause is the Prompts adaptive-reader migration (`3e8b104f6f` -> `f1275c8846`,
2026-08-25/26) plus `04e29673a2`'s canvas edit — **not asserted**, pending the swallowed traceback.
