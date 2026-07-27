---
id: TASK-1011
title: MCP risk tags are self-declared by the server being gated
status: To Do
assignee: []
created_date: '2026-07-27 20:27'
labels:
  - mcp
  - security
  - design
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A remote MCP tool's risk tags are derived from that server's own payload -- its risk_class field plus a free-form capabilities list, lowercased (MCP.hub_tool_catalog._extra_tags). Those tags are what resolve_effective_state uses to floor an inherited allow to ask. So a server can avoid the risk floor for its own tools simply by not declaring mutates, and can trip the floor for unrelated reasons by listing an ordinary word. The floor is a default rather than the only control -- users still set per-tool allow/ask/deny and a definition-hash guard catches later redefinition -- but the floor exists precisely to protect tools a user has not explicitly configured, which is exactly the case where the server's self-declaration is the only input. Surfaced while resolving TASK-845, which asked whether the network tag should move to the shared HIGH_RISK_TAGS and resolved no for this reason.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 It is decided and documented whether a server-supplied tag may lower a tool's risk classification,A tool whose server declares no tags is floored according to a policy we control rather than defaulting to unfloored,A test covers a server payload that omits mutates for a tool that clearly mutates,The decision is recorded at the definition site alongside the existing tag-set rationale
<!-- AC:END -->
