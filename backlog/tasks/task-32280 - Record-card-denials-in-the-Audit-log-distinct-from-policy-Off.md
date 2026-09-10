---
id: TASK-32280
title: 'Record card denials in the Audit log, distinct from policy Off'
status: To Do
assignee: []
created_date: '2026-09-10 19:12'
labels:
  - mcp
  - audit
  - approvals
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live: three approvals were logged and the user's Deny was not. The denial resolves at the pre-dispatch review hook and appears never to reach the provider's decision recorder (inferred). Where a record is written, policy-Off and user-Deny both use 'denied', so Audit cannot answer 'what did I refuse?'. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A card Deny (fast button or Submit) produces an execution-log row visible in Audit, Executions.
- [ ] #2 The Decision column and filter distinguish user denials from policy Off and from timeouts.
- [ ] #3 Tests cover the hook-level deny path end to end.
<!-- AC:END -->
