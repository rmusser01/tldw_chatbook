---
id: TASK-2902
title: Console — defer the hidden inspector rail and task surface past first paint
status: To Do
assignee: []
created_date: '2026-08-07 02:00'
labels:
  - console
  - performance
  - defer-past-first-paint
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Screen survey (task-2725 follow-up): the Console screen mounts 357 widgets and is the app's default tab, so its cost is also perceived cold-start cost. 124 widgets arrive hidden in three roots: `ConsoleInspectorRail#console-right-rail` (76), `ChatTaskCards#console-task-surface` (32), `CompactModelBar#console-compact-model-bar` (16) — ~35% deferrable.

This is deliberately LAST in the defer-past-first-paint series: `chat_screen.py` is the app's most complex screen and its sync pipeline (`_sync_native_console_chat_ui` and delegates) touches the rail, so the compose→load window audit is substantially harder than 2725/2900/2901. Do not start until both prior tasks have shipped and soaked. The audit must cover: every query of the three roots reachable from the sync path, `restore_state`, the control-bar build, and the session controllers introduced by console-decomposition wave 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [ ] Console first paint excludes the three hidden roots; rail/task-surface/model-bar all function once revealed.
- [ ] Console switch latency improves measurably live; cold-start-to-interactive improves.
- [ ] The full Console test surface stays green (including the worker-lifecycle and generation-actions suites).
- [ ] The compose→load window audit is recorded in the task notes (which query sites can run early, and why each is safe).
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->
