---
id: TASK-1011
title: MCP risk tags are self-declared by the server being gated
status: Done
assignee: []
created_date: '2026-07-27 20:27'
updated_date: '2026-07-27 20:42'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed as not-worth-the-complexity after analysing the actual resolution path. Recording the analysis so this is not re-discovered.

The concern as filed was that a remote MCP server supplies its own risk tags (MCP.hub_tool_catalog._extra_tags derives them from the server's risk_class field and its free-form capabilities list), so a server could avoid the risk floor for its own tools by declaring nothing. That is true but much narrower than it sounds, for two reasons found by reading resolve_effective_state:

1. DEFAULT_GLOBAL is 'ask'. Out of the box every unconfigured MCP tool prompts regardless of its tags, so the floor is not what protects a default install -- the default state is.

2. The floor only applies when origin != 'tool_override'. An explicit per-tool allow is deliberately never floored, on the documented reasoning that the operator opted in with knowledge of that specific tool.

So the exposure is limited to a user who has broadened permissions -- set global_default: allow, or a server_default: allow -- and is then relying on a safety net woven by the server itself to re-tighten individual tools. That user has already extended trust to the server at the server level; a tag they cannot verify is a weak addition to a decision they made deliberately.

Fixing it properly would mean either treating absent tags as unknown-rather-than-safe (which floors everything for allow-configured servers, defeating the point of choosing allow) or classifying arbitrary remote tools ourselves from name and schema, which is a hard problem with a poor accuracy ceiling and its own false-positive cost.

Related and unaffected: TASK-845 resolved that 'network' stays in BUILTIN_HIGH_RISK_TAGS rather than the shared set, precisely because MCP tags are server-supplied. That decision stands on the same provenance fact and does not depend on this one.
<!-- SECTION:NOTES:END -->
