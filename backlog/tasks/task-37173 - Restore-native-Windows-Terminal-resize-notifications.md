---
id: TASK-37173
title: Restore native Windows Terminal resize notifications
status: Done
assignee:
  - '@codex'
created_date: '2026-10-28 18:46'
updated_date: '2026-10-28 19:00'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resizing or maximizing Windows Terminal must resize the active Chatbook screen and keep its footer at the bottom.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native Windows Terminal before-and-after evidence demonstrates resize recovery.
- [x] #2 Windows input retains virtual terminal input and enables resize notifications after each driver start; original console mode is restored on close.
- [x] #3 Explicit custom drivers and non-Windows behavior are preserved; targeted tests and lint pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Routine compatibility bug fix within the existing TextualAppGuards boundary; no dependency or runtime contract changes. ADR-097 boot budgets remain unchanged.
1. Capture native before sizes and add a failing Windows driver regression.
2. Select a narrowly scoped Windows driver adapter through the existing TextualAppGuards seam; preserve custom drivers and upstream cleanup.
3. Run targeted tests and native after resize verification, then document measured evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Enabled ENABLE_WINDOW_INPUT through TextualAppGuards.get_driver_class after every default WindowsDriver start. Preserved VT flags, custom drivers, Unix behavior and upstream console restoration. Added driver restart/cleanup/custom-selection regressions. Marked the existing real-app wiring test bootstrap_profile after reproducing its config-selection failure with the fix disabled. Native side-by-side Logs test: both OS panes grew from 98x42 to 135x68; stock layout stayed 98x42, fixed layout filled 135x68 with footer on row 67. User confirmed only the fixed pane resized correctly. Targeted tests: 16 passed, 1 Unix-only skip; Ruff check and format passed; independent review found no actionable issues. Docs: Docs/Architecture/app-shell.md, Docs/superpowers/plans/2026-10-28-windows-terminal-resize.md and qa/windows-terminal-resize-2026-10-28/README.md. Added the TERM=dumb QA-launcher lesson. ADR required: no (routine compatibility repair; ADR-097 budgets preserved).
<!-- SECTION:NOTES:END -->
