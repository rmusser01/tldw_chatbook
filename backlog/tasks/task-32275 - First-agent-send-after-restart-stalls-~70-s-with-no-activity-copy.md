---
id: TASK-32275
title: First agent send after restart stalls ~70 s with no activity copy
status: To Do
assignee: []
created_date: '2026-09-10 19:10'
labels:
  - console
  - agents
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a warm profile, the first Console agent send after an app restart showed an empty assistant row and 'Run: Agent running.' for about 70 seconds before the provider was called; a later send in the same instance took about 5 seconds. Nothing on screen said what the app was waiting on. The built-in MCP server spawn/discovery is the suspected cause; not traced. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the delay is identified and either removed or bounded by a visible timeout with a stated reason.
- [ ] #2 While pre-provider setup runs, the assistant row or status strip says what is happening (for example connecting tools) instead of staying blank.
- [ ] #3 A diagnostic event or test pins the pre-provider setup time budget.
<!-- AC:END -->
