---
id: TASK-32281
title: >-
  Exact-input allow rules: review and remove UI, honoured on every row that
  offers it
status: To Do
assignee: []
created_date: '2026-09-10 19:13'
labels:
  - mcp
  - permissions
  - approvals
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'Always allow this exact input' persists argument rules that no UI lists or removes (no references under UI/ or Widgets/), and Virtual CLI rows offer it while their verdict path recognises only once, session, always and deny. User decision 2026-09-10: keep the option on the card and build the rules list. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The tool inspector lists each stored exact-input allow with its arguments and a Remove action, and removing it makes the next matching call ask again.
- [x] #2 The Permissions matrix marks tools that carry argument rules.
- [x] #3 Every producer that offers the exact-input decision honours it (Virtual CLI included) or narrows its options so it is not offered.
- [x] #4 The user guide describes the rule and where to remove it.
<!-- AC:END -->

## Implementation Plan

1. Trace the arg-rule storage shape (`MCPPermissionStore.add_tool_arg_rule`) and add paired `list_tool_arg_rules`/`remove_tool_arg_rule` methods, mirrored on `UnifiedMCPControlPlaneService`.
2. Render one row per rule in the inspector's permission block (`_render_permission_container`), threaded through every real `show_tool()`/`show_permission()` call site, with a Remove button posting a new `RemoveArgRuleRequested` message; wire the workbench handler to delete + resync.
3. Add a `≡` marker to the Permissions matrix's State cell for a tool carrying rules (both the MCP and built-in matrix sections), and document it in the legend.
4. Decide honour-vs-narrow for Virtual CLI and fix `_ask_verdict()` accordingly; wire persistence/consultation through `console_chat_controller.py`'s composition.
5. Fix a latent strict-store-validation gap the AC#2 integration test surfaced (see notes).
6. Document the rule in the user guide (AC#4).

## Implementation Notes

See `backlog task edit 32281 --notes` for the final summary (this CLI call replaces this section).
