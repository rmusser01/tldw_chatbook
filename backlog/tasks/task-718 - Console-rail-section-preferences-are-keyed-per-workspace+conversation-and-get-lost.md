---
id: TASK-718
title: Console rail section preferences are keyed per workspace+conversation and get lost
status: To Do
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - console
  - config
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
[console.rail_state] keys embed both workspace id and conversation id, so section open/closed preferences reset with every new conversation and a Details toggle made moments earlier is lost after switching workspaces and back (live-verified; observed keys also paired one conversation id with two different workspaces). If per-workspace layout memory is the intent, the conversation component defeats it. Finding M7.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rail section open/closed preferences survive switching away from and back to a workspace
- [ ] #2 The persistence key strategy is documented and does not multiply entries per conversation
- [ ] #3 Existing stale keys are pruned or migrated
<!-- AC:END -->
