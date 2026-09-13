---
id: TASK-22504
title: >-
  Defer console_voice_input off the Chat first-paint import leg
status: To Do
assignee: []
created_date: '2026-08-26'
labels:
  - performance
  - startup
  - console
priority: low
dependencies: []
---

## Description

Source: close-out of the 2026-08-24 holistic performance review's burn-down (29 tasks,
TASK-22200..22228, all merged 2026-08-25/26). Evidence: `Docs/Design/2026-08-24-holistic-perf-review.md` plus the originating task's
Implementation Notes.

Left explicitly unfinished by TASK-22213 with its evidence: `Chat/console_voice_input`
(2,260 LOC) rides the Chat first-paint leg, and `chat_screen.py:241` is NOT the load-bearing
edge — `composer_bar.py:39`, `dictation.py:120` and `hands_free.py:124` all module-import it
too, so deferring needs its own task across those three seams.

22213's census-at-`_ui_ready` guard (budget 970, measured 938-941) is the instrument that
will show the win or its absence.

## Acceptance Criteria

- [ ] `console_voice_input` and its transitive closure are absent from `sys.modules` at `_ui_ready` on a warm boot, or the residual is attributed and stated
- [ ] Dictation, hands-free and composer voice wiring still work at first use (their suites green; a live check for the first voice interaction)
- [ ] The `_ui_ready` census budget is lowered to the new measurement so the win is pinned
