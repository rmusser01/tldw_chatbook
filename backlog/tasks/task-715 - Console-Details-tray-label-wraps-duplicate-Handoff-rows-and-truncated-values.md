---
id: TASK-715
title: Console Details tray label wraps duplicate Handoff rows and truncated values
status: Done
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - console
  - workspaces
  - copy
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At rail width the Server handoff label wraps leaving an orphaned lowercase 'handoff' line; two different rows are both labeled Handoff (package list vs ACP status); values truncate ('Off in Default work…', 'ACP handoff: Not co…'); and the tray presents jargon-dense rows (handoff package, ACP task/run package, audit) for features that have no production writer today (server/sync/runtime/ACP states are code-verified unreachable). Finding M4; captures cap-10, cap-18, cap-19.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No label in the Details tray wraps mid-phrase at the rail's real width
- [x] #2 Each Details row has a unique label distinguishing package handoff from ACP handoff
- [x] #3 Truncated values expose their full text (tooltip or wrap-by-design)
- [x] #4 Rows whose backing feature cannot be configured anywhere in the UI are hidden or collapsed behind a single not-configured line
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red tests mounting ConsoleWorkspaceDetailsTray: collapsed default, unique fitting labels when configured, non-truncating file-tools value.
2. Fix labels/values + collapse logic in console_workspace_details.py; update pinned assertions across rail/native-flow suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root causes: the StatusPair label column is 12 cells, so "Server handoff" (14) wrapped into an orphaned "handoff" line; the ACP row was folded under the label "Handoff", duplicating the Handoff section title; "Off in Default workspace" ellipsized in the ~23-cell value column. Fixes in console_workspace_details.py: labels now "Server" / "ACP" (fit, unique), file-tools default value "Off in Default", removed the Handoff label-fold in _status_pair. New collapse: when sync + server handoff + ACP are ALL factory defaults (none has a production writer today) the five jargon rows collapse into one line "Server features (sync, handoff, ACP): not configured. Chats stay local." (#console-workspace-server-features-collapsed); any real state brings the full rows back. Values keep the existing nowrap+ellipsis+tooltip (TASK-384). New suite Tests/UI/test_console_workspace_details_tray.py (3 tests); updated pinned assertions in test_console_workspace_context_rail.py, test_console_rail_sections.py, test_console_native_chat_flow.py (201 passing).
<!-- SECTION:NOTES:END -->
