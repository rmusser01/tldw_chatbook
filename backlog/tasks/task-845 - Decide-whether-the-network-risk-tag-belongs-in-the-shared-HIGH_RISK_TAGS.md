---
id: TASK-845
title: Decide whether the network risk tag belongs in the shared HIGH_RISK_TAGS
status: To Do
assignee: []
created_date: '2026-07-27 03:46'
labels:
  - tools
  - security
  - mcp
  - decision
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #953 added the `network` risk tag to `BUILTIN_HIGH_RISK_TAGS` rather than the shared `HIGH_RISK_TAGS`, so an egress-capable in-process built-in floors to ask while MCP resolution is untouched. This reverses an earlier operator decision to widen the shared set. The reversal was driven by dev's own comment on `BUILTIN_HIGH_RISK_TAGS`, which states that widening `HIGH_RISK_TAGS` would make remote MCP tools carrying that tag start prompting and that doing so is deliberately not the current phase's call. Both readings are defensible: the shared set closes the exfiltration leg for every provider, the built-in-only set avoids changing behaviour for users' existing MCP servers without their asking. The change is one line either way; what is needed is a decision on record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The intended scope of the network tag is decided and written down,The tag sits in whichever set that decision names,A comment at the definition site states why that set and not the other
<!-- AC:END -->
