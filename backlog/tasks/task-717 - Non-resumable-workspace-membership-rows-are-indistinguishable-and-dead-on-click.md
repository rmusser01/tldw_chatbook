---
id: TASK-717
title: Non-resumable workspace membership rows are indistinguishable and dead on click
status: Done
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - console
  - workspaces
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Membership rows that cannot be resumed (role source, or conversation records missing from the chat DB) render identically to openable conversation rows and produce no reaction at all when clicked - no toast, no navigation. Live-verified with a ghost membership (cap-26). Finding M6.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rows that cannot be opened are visually distinct from openable conversation rows
- [x] #2 Clicking a non-resumable row produces visible feedback explaining why nothing opened
- [x] #3 A membership whose conversation record is missing surfaces a recovery hint that matches an affordance that actually exists
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red test: ghost membership row press -> single honest toast, row marked broken+disabled.
2. Rework resume return contract (True/None-transient/False-missing), caller-owned missing-record UX, lazy broken marking threaded through browser rows, dim CSS.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Probe correction to the UAT report: clicking a ghost row DID produce feedback at the code level - two stacked toasts, one promising a Library affordance that does not exist ("Open this saved conversation from Library before switching here." after "Saved conversation was not found."). Fix: `_resume_console_workspace_conversation` now returns True / None (transient failure, already notified) / False (record missing, caller owns UX); both callers (rail row press + Ctrl+K switcher) show ONE honest toast ("could not be loaded - its record is missing") and call `_mark_console_conversation_row_broken`, which records the id and resyncs the rail. Openability cannot be known at render time without a per-row DB probe, so rows are marked lazily after the first informative failure: `openable` field threaded through ConsoleConversationBrowserInputRow/Row, membership row build checks the broken set, and the tray renders non-openable rows disabled + dim italic (`.console-workspace-conversation-row-broken`) with an explanatory tooltip. New test Tests/UI/test_console_workspace_dead_rows.py; 57 workspace-suite tests green.
<!-- SECTION:NOTES:END -->
