---
id: TASK-761
title: Restore Console agent dictionary send integration baseline
status: To Do
assignee: []
created_date: '2026-07-26 17:57'
labels:
  - console
  - chat-dictionaries
  - baseline
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the Console agent send integration contract so a conversation dictionary is applied before the agent bridge receives provider messages, eliminating the deterministic failure inherited from dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Agent-path Console sends apply the active conversation dictionary before agent dispatch,Provider-path dictionary behavior remains unchanged,The exact agent dictionary integration regression passes offline,Focused Console controller and dictionary tests pass
<!-- AC:END -->
