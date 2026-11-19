---
id: TASK-42016
title: "TASK-40714's draft-snapshot sends are untested with the agent runtime on"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- console
- tests
priority: low
references:
- Tests/UI/test_console_send_without_mcp_store.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41900 (round 17). TASK-41900 runs TASK-40714's ordinary send with the agent runtime on; the draft-snapshot sends (queued, armed tokenless, spoken) go through the same `_AgentSources` root but are still tested with the runtime off.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each draft-snapshot send is also tested with the agent runtime on and an unbound store
<!-- AC:END -->
