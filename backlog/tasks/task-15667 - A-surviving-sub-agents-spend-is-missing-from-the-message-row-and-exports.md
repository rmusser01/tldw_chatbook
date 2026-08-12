---
id: TASK-15667
title: 'A surviving sub-agent''s spend is missing from the message row and exports'
status: To Do
assignee: []
created_date: '2026-08-11 21:30'
labels:
  - console
  - agents
  - cost
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 3a-1 delivered audit F3 as OBSERVABLE, not fixed: a survivor's post-turn token spend reaches the Console cost chip's tooltip (`Sub-agents: N tok (not priced)`) and the chip's own token total, and nothing else. It is absent from the assistant message's stored usage row and from conversation exports, and it is remembered only for the lifetime of the controller instance - close the Console screen and it is gone. This is the partiality that remains after F3; the full re-attach is tracked separately and depends on a signal PR 3a-2 builds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A conversation export accounts for sub-agent spend, or states explicitly that it excludes it
- [ ] #2 The assistant message row's usage reflects the sub-agents that ran underneath that turn
- [ ] #3 The figure survives closing and reopening the Console screen
- [ ] #4 The User Guide's honest-limits paragraph is updated when the limit no longer holds
<!-- AC:END -->
