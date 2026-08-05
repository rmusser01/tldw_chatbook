---
id: TASK-845
title: Decide whether the network risk tag belongs in the shared HIGH_RISK_TAGS
status: Done
assignee: []
created_date: '2026-07-27 03:46'
updated_date: '2026-07-27 20:27'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved NO on evidence: network stays in BUILTIN_HIGH_RISK_TAGS and does not move to the shared HIGH_RISK_TAGS. An MCP tool's tags are not ours -- they are derived from the remote server's own risk_class and free-form capabilities list, lowercased (MCP.hub_tool_catalog._extra_tags). 'network' is an ordinary word for a server to list among its capabilities, so widening the shared set would not be the no-op it appears to be; it would start prompting on real servers because of a string chosen for unrelated reasons. The built-in set is a vocabulary we control and can reason about; the shared set is partly server-supplied and should stay narrow. Rationale recorded at the definition site. This reverses the earlier preference for widening the shared set, and supersedes it with the provenance evidence that was not available when that preference was expressed. The investigation also surfaced a larger concern -- that MCP risk tags are self-declared by the very server being gated -- filed separately.
<!-- SECTION:NOTES:END -->
