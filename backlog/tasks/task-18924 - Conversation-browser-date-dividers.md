---
id: TASK-18924
title: 'Console conversation browser: date dividers'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - console
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's sidebar date dividers (2026-08-19 hermes-release review). The Console left-rail Conversations browser groups rows only as Starred / Workspaces / Chats. Within the "Chats" group (composition decided in implementation), bucket conversation rows under date headers — Today / Yesterday / This week / Older — computed from last-activity time, so a long list reads chronologically at a glance. Pure polish over the existing browser state (Workspaces/conversation_browser_state.py).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Conversation rows render under date-bucket headers computed from last-activity time, with deterministic bucket boundaries (today/yesterday/this-week/older) stable in tests
- [ ] #2 Search filtering remains authoritative: the pinned behavior documents whether dividers stay or flatten while filtering, and tests pin that choice
- [ ] #3 Empty buckets render nothing; the existing group collapse behavior and row secondary lines are unchanged
- [ ] #4 UI tests cover bucket boundaries, search interplay, and no regression to Starred/Workspaces/Chats grouping
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: display-only grouping over existing browser state; no schema or boundary change.

1. Compute date buckets in the browser state layer from conversation last-activity
2. Render dividers in the tray's Chats group (decide and pin search interplay)
3. UI tests + sessions-tabs-workspaces.md note
<!-- SECTION:PLAN:END -->
