---
id: TASK-32273
title: >-
  Tool-call continuation crashes on mappingproxy JSON and is reported as a
  provider HTTP 400
status: To Do
assignee: []
created_date: '2026-09-10 19:09'
labels:
  - agents
  - llm-calls
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After a native tool call, the next provider request carries a MappingProxyType (ToolCall.arguments became immutable in 4ed4757d53, 2026-09-04) and the requests-based OpenAI-compatible handler cannot serialise it. The failure is shown to the user as 'provider returned HTTP 400 ... The provider rejected this request. Confirm the model is still available', blaming the provider for a client-side bug. Reproduced live with the custom provider; other handlers unverified. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A multi-step tool-call turn (find_tools, load_tools, tool) completes through the custom/OpenAI-compatible handler without a serialisation error.
- [ ] #2 Every provider handler that re-sends tool-call history serialises immutable argument mappings; a regression test covers the serialisation seam.
- [ ] #3 A client-side serialisation failure is reported to the user as an app error, never as a provider rejection or HTTP status.
<!-- AC:END -->
