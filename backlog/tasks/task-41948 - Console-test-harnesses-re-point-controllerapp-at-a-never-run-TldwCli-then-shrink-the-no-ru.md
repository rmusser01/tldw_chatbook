---
id: TASK-41948
title: "Console test harnesses re-point `controller.app` at a never-run TldwCli; then shrink the no-running-app guard"
status: To Do
assignee: []
updated_date: '2026-11-23 07:40'
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI
- tldw_chatbook/Chat/console_interrupt_rounds.py
- backlog/decisions/314-indefinite-human-approval-waits.md
dependencies:
- TASK-41637
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41637's guard fails a round closed only when the App is not running AND no Console view can answer, because harnesses (e.g. `ProductionConsoleHarness`) point `controller.app` at a never-run `TldwCli` next to a live view (about 6 nodes depend on it). If the harnesses pointed at the running host, the guard could be just `not app.is_running` (with the AttributeError fallback).

2026-11-23 (round-22 filing): TASK-42465 proposes the other fix for the same guard: judge the arm-time liveness check by the attached view's own App (`view.app.is_running`) instead of repointing the harnesses. Its AC#1 asks the owner to choose between the two; if this task's harness fix is chosen, TASK-42465 closes as covered here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The harnesses point `controller.app` at the running host
- [ ] #2 The guard drops the view condition; ADR-314's clarification is updated; the round and harness tests pass
<!-- AC:END -->
