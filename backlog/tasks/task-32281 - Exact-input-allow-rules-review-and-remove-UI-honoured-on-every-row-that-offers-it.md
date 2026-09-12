---
id: TASK-32281
title: >-
  Exact-input allow rules: review and remove UI, honoured on every row that
  offers it
status: Done
assignee: []
created_date: '2026-09-10 19:13'
updated_date: '2026-09-11 06:31'
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

<!-- SECTION:PLAN:BEGIN -->
1. Trace the arg-rule storage shape (`MCPPermissionStore.add_tool_arg_rule`) and add paired `list_tool_arg_rules`/`remove_tool_arg_rule` methods, mirrored on `UnifiedMCPControlPlaneService`.
2. Render one row per rule in the inspector's permission block (`_render_permission_container`), threaded through every real `show_tool()`/`show_permission()` call site, with a Remove button posting a new `RemoveArgRuleRequested` message; wire the workbench handler to delete + resync.
3. Add a `≡` marker to the Permissions matrix's State cell for a tool carrying rules (both the MCP and built-in matrix sections), and document it in the legend.
4. Decide honour-vs-narrow for Virtual CLI and fix `_ask_verdict()` accordingly; wire persistence/consultation through `console_chat_controller.py`'s composition.
5. Fix a latent strict-store-validation gap the AC#2 integration test surfaced (see notes).
6. Document the rule in the user guide (AC#4).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added MCPPermissionStore.list_tool_arg_rules()/remove_tool_arg_rule() (mirrored on UnifiedMCPControlPlaneService), and fixed a latent gap in _validate_strict_profile() that the round-trip test surfaced: a tool entry with only arg_rules (no whole-tool state) has always failed strict-snapshot validation since add_tool_arg_rule shipped, invalidating the WHOLE profile for read_snapshot_strict()/read_profile_inventory_snapshot() callers (the Permissions-mode profile selector) the moment any tool got an exact-input rule with no state override. The inspector's permission block now renders one row per rule (capped 60-char args summary) with a Remove button, threaded through every show_tool()/show_permission() call site via a new arg_rules kwarg and MCPWorkbench._arg_rules_for_row(); a new RemoveArgRuleRequested message deletes the rule and resyncs. The Permissions matrix (both MCP and built-in sections) appends a ≡ marker to a tool's State cell when it carries a rule, via MCPWorkbench._tool_has_arg_rules() reading the already-loaded servers_payload (accepting both list and tuple, since the frozen-snapshot path turns lists into tuples); the legend gained '≡ exact-input allows'. For Virtual CLI (the one broken producer named in the task): chose HONOUR over narrow, since its argument shape -- a fixed command enum plus a bounded argv array -- is stable; _ask_verdict() now persists the rule on both the stamped-decision and live-callback paths (both previously mishandled allow_matching: one silently re-asked, the other actively denied the call), and pending_gate_for()/invoke() now consult arg_rule_allows so a matching next call resolves allow without re-asking, wired through console_chat_controller.py's _compose_virtual_cli_provider. Every other MCPPendingCall producer (LocalToolProvider, raw_shell_tool_provider, agent_lesson_promotion, console_chat_controller's own two sites) already narrows its options to exclude allow_matching -- confirmed via grep, none needed touching. User guide: new 'Exact-input allow rules' section in mcp.md (with a Verified-against stamp), and the fifth option named plus cross-linked in console/agent-runs-and-tools.md. Tests: 8 new in Tests/MCP/test_permission_store.py (list/remove round trip, strict-validator regression), 6 new across Tests/UI/test_mcp_workbench.py and Tests/UI/test_mcp_inspector.py (marker + row rendering + Remove button + end-to-end store round trip), 3 new in Tests/Agents/test_virtual_cli_provider.py, 1 new in Tests/Chat/test_console_virtual_cli_approval.py; full runs of all touched files are green except two PRE-EXISTING failures in test_console_virtual_cli_approval.py (a turn_context.tool_policy_profile_id AttributeError, confirmed present at HEAD before this change, not this task's to fix). preflight.sh is green (regenerated production-diagnostic-inventory.json for two new safe warning-log call sites). Whole-branch review round (R22, commit b1d420cba2) found the exact-input rules INERT on high-risk tools while still being advertised: permission_store.arg_rule_allows returns False outright when set(tool.tags) & HIGH_RISK_TAGS, so an allow_matching rule on a mutates/process tool can never quiet a call. Three changes. (a) MCPToolProvider narrows such a row's card options to approve_once/approve_session/always_allow/deny via a new _options_for_tool() applied at BOTH MCPPendingCall construction sites (pending_gate_for and invoke's own fallback card). always_allow deliberately STAYS: it persists a tool-level allow whose origin is tool_override, the one origin resolve_effective_state's risk floor explicitly spares, so unlike the argument rule it really does take effect. (b) _apply_verdict's allow_matching branch degrades to a one-time approval and logs at DEBUG when the tool is high-risk, rather than persisting permanently dead state -- defense in depth for a verdict that arrives from a stale card or a caller ignoring options. (c) The inspector renders such a rule as 'Exact-input allow (not in effect: risk floor) - <redacted args>', Remove button kept so a stale rule can still be cleared. The Permissions matrix's ≡ marker is unchanged: it means 'has a rule', which stays true. Note for the reader: with today's catalog builders (hub_tool_catalog.local_tools_from_record/builtin_tools_from_inventory both hard-code tags=()), only server-source tools carry tags at all, so (a)/(b) are correctness hardening the provider's present catalog cannot yet exercise, while (c) is reachable now through the hub's server-source rows. Tests: test_high_risk_rows_do_not_offer_the_inert_exact_input_option and test_allow_matching_on_a_high_risk_tool_approves_once_without_persisting in Tests/Agents/test_mcp_tool_provider.py; test_arg_rule_row_on_a_high_risk_tool_says_it_is_not_in_effect in Tests/UI/test_mcp_inspector.py (row text plus a real Remove click still posting RemoveArgRuleRequested). Docs: the 'Exact-input allow rules' section in mcp.md gains the risk-floor caveat and a fix-round stamp. Also in the same wave (commit aa95995aad), _tool_entry's docstring stopped claiming remove_tool_arg_rule shares it -- that walks inline because it also needs the parent tools mapping to drop an emptied entry. Files: tldw_chatbook/Agents/mcp_tool_provider.py, tldw_chatbook/UI/MCP_Modules/mcp_inspector.py, tldw_chatbook/MCP/permission_store.py, Docs/User_Guide/mcp.md, Docs/security/production-diagnostic-inventory.json (one new DEBUG line, reviewed: interpolates tool.server_key/tool.name only), plus the two test files.
<!-- SECTION:NOTES:END -->
