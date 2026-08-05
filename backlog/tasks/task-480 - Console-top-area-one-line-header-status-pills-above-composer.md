---
id: TASK-480
title: 'Console top area: one-line header + status pills above composer'
status: Done
assignee: []
created_date: '2026-07-22 03:05'
updated_date: '2026-07-22 05:39'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Collapse the Console 3-line identity header into one full-width line and move the status pills from the top control bar to a full-width strip directly above the composer. Spec: Docs/superpowers/specs/2026-07-21-console-top-area-layout-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console header renders on a single row with title, subtitle, and Ready badge
- [x] #2 Subtitle ellipsizes when narrow and the Ready badge stays flush to the right edge
- [x] #3 Status pills render in a full-width strip directly above the composer
- [x] #4 The action row (New tab/Settings/...) stays at the top under the header
- [x] #5 No other screen's header changes and all chip ids are preserved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-21-console-top-area-layout.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reworked the Console screen's top area: (1) the 3-line identity header collapses to one full-width inline row; (2) the status pills moved from the top control bar to a full-width strip directly above the composer.

Approach: (1) Header — a Console-scoped 'console-header-inline' class flips the shared DestinationHeader instance to layout:horizontal with an ellipsizing 1fr subtitle and a flush-right Ready badge; all rules id-prefixed (#console-workbench-header.console-header-inline) to beat the id-only and density rules; border:none for the one-row box; em-dash prepended to the Console-only subtitle. Shared DestinationHeader Python untouched. (2) Pills — extracted a new ConsoleStatusChips widget (own file) owning all the pills + sync_state + the approvals-review action; ConsoleControlBar slimmed to actions-only (height 2->1, token $ds-console-control-bar-height 2->1); chips composed as a full-width strip above the composer; _sync_console_control_bar syncs the strip. DEVIATION (user-approved): dev had added an 8th 'Scope' chip (task-10) after the spec was written; per user decision it moved down WITH the pills into ConsoleStatusChips (ConsoleScopeChip + sync_scope_chip + scope_state relocated; chat_screen import/sync retargeted; @on(OpenRequested) stays on the screen via message bubbling).

Verification: header geometry via a bundled-CSS test App (harness doesn't load the bundle); ConsoleHarness test for class+subtitle; narrow-width ellipsize test; chip unit tests; coupling-test split; placement test; 205/205 across the 4 affected suites; full Tests/UI + Tests/Chat sweep = zero novel regressions vs origin/dev (Console-adjacent visual failures identical at merge-base); opus whole-branch review Ready-to-merge; live tmux/SVG wide+narrow verification.

Files: console_status_chips.py (new), console_control_bar.py, chat_screen.py, console_workbench_state.py, console_retrieval_scope_row.py, css/components/_agentic_terminal.tcss + core/_variables.tcss (+ regenerated bundle), test_console_workbench_contract.py, test_console_status_chips.py (new), test_console_scope_row.py.
<!-- SECTION:NOTES:END -->
