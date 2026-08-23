---
id: TASK-21162
title: Add MCP fewer permission prompts recommendations
status: Done
assignee:
  - '@codex'
created_date: '2026-08-01 21:58'
updated_date: '2026-08-23 17:00'
labels:
  - mcp
  - console
  - permissions
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help Chatbook users reduce repeated MCP permission prompts by analyzing the local MCP execution log and recommending safe, reviewable tool-level permission changes without telemetry or model-based auto-approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User can run a Chatbook-native fewer-permission-prompts command and get a local MCP recommendation report.
- [x] #2 Report uses the local redacted MCP execution log and existing permission store without adding telemetry or tracking.
- [x] #3 Recommendations only target repeated approved MCP tool calls that are still ask-gated and are not blocked by existing safety downgrades.
- [x] #4 User can review recommended tool-level allow changes through existing MCP permission APIs instead of hand-editing JSON.
- [x] #5 Auto mode and bash command allowlisting are explicitly deferred.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/081-mcp-prompt-reduction-recommendations.md
Reason: This changes local permission recommendation behavior, persisted security-state workflows, privacy boundaries, and Console/MCP service contracts.

1. Add a pure MCP prompt-reduction analyzer that derives recommendations from recent execution-log rows, live `HubTool` definitions, and resolved `EffectiveToolState` values.
2. Add `UnifiedMCPControlPlaneService` report/apply methods that collect local MCP tools, read the bounded execution log, and persist recommended allows through `set_tool_state(..., "allow", tool=tool)`.
3. Register `/fewer-permission-prompts` as a Console built-in command and render a compact local report as a Console system message.
4. Cover the analyzer, service integration, and command grammar with focused tests, then run `git diff --check`.

Detailed plan: `Docs/superpowers/plans/2026-08-01-mcp-fewer-permission-prompts.md`
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-081's local-only MCP recommendation flow. The new analyzer derives recommendations from the redacted MCP execution log, live `HubTool` catalog entries, and effective permission state, while excluding already allowed/denied tools, stale or missing tools, definition changes, high-risk floors, and below-threshold approvals.

Added `UnifiedMCPControlPlaneService.permission_prompt_recommendations()` and `apply_permission_prompt_recommendation()` so reports and allow changes go through the existing MCP permission store and definition-hash path. Report validation and application share one live catalog snapshot, preventing a changed definition from inheriting an older recommendation. Registered `/fewer-permission-prompts` in the native Console command grammar and rendered a compact Console system report that explicitly defers Auto Mode and bash allowlisting while explaining empty states and safety exclusions.

Executions covered by an existing session grant are now recorded as `approved-session`, distinct from the prompted `approved` decision, so cached executions cannot inflate the recommendation count. The MCP Audit filter exposes both decisions.

Updated tests for the pure analyzer and formatter, control-plane service integration, Console grammar and handler, command completion, session-approval audit behavior, audit filtering, and reserved command-name drift. Per user direction, final verification was scoped to changed functionality: `PYTEST_DEBUG_TEMPROOT=/tmp/tldw_feature_pytest_remote PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/tldw_test_stubs:$PWD .venv/bin/python -m pytest Tests/MCP/test_permission_prompt_reducer.py Tests/MCP/test_control_plane_prompt_reducer.py Tests/UI/test_console_fewer_permission_prompts_command.py Tests/Chat/test_console_command_grammar.py Tests/UI/test_console_command_composer.py Tests/Agents/test_mcp_tool_provider.py::test_invoke_ask_callback_approve_session_persists_and_short_circuits_next_call Tests/Agents/test_mcp_tool_provider.py::test_invoke_stamped_approve_session_counts_one_prompt_for_same_name_calls Tests/UI/test_mcp_audit_mode.py::test_select_options_cover_full_decision_and_initiator_vocabulary Tests/Library/test_library_skills_state.py::test_shadow_name_set_stays_in_sync_with_real_sources -q --tb=short` passed with 90 tests and 2 dependency warnings. `git diff --check` also passed.

Modified files: `tldw_chatbook/MCP/permission_prompt_reducer.py`, `tldw_chatbook/MCP/unified_control_plane_service.py`, `tldw_chatbook/Agents/mcp_tool_provider.py`, `tldw_chatbook/Chat/console_command_grammar.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/UI/MCP_Modules/mcp_audit_mode.py`, focused tests, ADR-081, ADR index, and the Superpowers implementation plan.
<!-- SECTION:NOTES:END -->
