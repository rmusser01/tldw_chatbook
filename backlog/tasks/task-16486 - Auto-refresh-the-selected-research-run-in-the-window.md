---
id: TASK-16486
title: Auto-refresh the selected research run in the window
status: Done
assignee:
  - '@robert'
created_date: '2026-08-16 03:39'
updated_date: '2026-08-16 03:47'
labels:
  - research
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase and progress events stream into research_run_events, but the window only shows them after manual actions - a running run's status goes stale in the detail pane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A mounted interval refreshes the selected local run while it is non-terminal without disturbing payload state (bundle or artifact selections),Terminal runs and server-source selections stop refreshing,Tests cover the refresh path, the non-terminal guard, and payload-state preservation
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `on_mount` sets a 2s interval calling `_auto_refresh_selected_run`: refreshes the selected LOCAL run's detail while its status is non-terminal (completed/failed/cancelled/draft skip), preserving payload state (bundle/artifact selections untouched -- only `selected_run` + detail render update). Controller errors are swallowed silently (a dead scope service must not spam the status line every 2s). Server-source selections skip (server observation has its own streaming surface via Watch Events).
- Tests: refresh updates phase + preserves bundle; terminal and server-source guards verified silent (controller never called).
<!-- SECTION:NOTES:END -->
