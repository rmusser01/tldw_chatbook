---
id: TASK-32285
title: 'Permissions mode polish: stale preview count, clipped legend, kill-switch hint'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 19:15'
updated_date: '2026-09-11 04:31'
labels:
  - mcp
  - permissions
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After Space set fs_read to Allow the preview still read '0 allow, 30 ask, 0 off'; the legend and gate breadcrumb clip mid-sentence at 50 rows; the kill-switch hint names only calculator and date/time; the kill switch has two differently worded refusal strings. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The preview counts update on every state change.
- [x] #2 The legend and gate breadcrumb are fully readable at 50 rows.
- [x] #3 The kill-switch hint describes its real blast radius and one refusal wording is used everywhere.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the preview/echo recompute path in mcp_workbench.py and the legend/hint layout in mcp_permissions_mode.py; verify with live-widget probes whether the reported staleness/clipping still reproduce on this branch's HEAD.\n2. Write regression tests (TDD) locking in whichever behaviors are already correct, plus RED tests for the two real gaps found (kill-switch hint wording, refusal-string wording).\n3. Implement the two real fixes: reword the kill-switch hint, and unify the four kill-switch refusal constants (plus the builtin gate's own inline copy) to one shared sentence.\n4. Re-run the touched test files plus the wider refusal/permissions suites to confirm green and rule out regressions.\n5. Update the task file and close it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Traced the preview/echo path in mcp_workbench.py (_build_permission_preview, _sync_permissions_mode) and the legend/hint layout in mcp_permissions_mode.py, then verified with live-widget probes and new regression tests against real store/service fakes that AC#1 (preview counts) and AC#2 (legend/breadcrumb wrapping at 100 columns, height:auto already set) were already correct on this branch's HEAD (f2c153622a) -- the live evidence's staleness/clipping had already been fixed by tasks 13-17 landing on this branch. Locked both in with new regression tests (test_preview_counts_recompute_on_every_cycle_not_cached in Tests/UI/test_mcp_workbench.py with a synthetic 30-tool server matching the live evidence's scale, and test_legend_and_breadcrumb_are_fully_readable_at_100_columns in Tests/UI/test_mcp_permissions_mode.py under the real bundled CSS harness) rather than touching working code. AC#3 needed real fixes: reworded the kill-switch hint Static in mcp_permissions_mode.py's compose() from 'Also disables built-in tools (calculator, date/time).' to 'Also blocks the app's own built-in tools (calculator, date/time, file and note tools).' (tightened the existing pinned test to assert the exact string), and unified the four differently-worded kill-switch refusal strings (console_chat_controller.KILL_SWITCH_REFUSAL, mcp_tool_provider.KILL_SWITCH_REFUSAL, local_tool_provider.LOCAL_KILL_SWITCH_REFUSAL, and console_agent_bridge._BUILTIN_KILL_SWITCH_REFUSAL's hand-copy of builtin_tool_gate.BuiltinToolGate.check()'s inline literal) to one shared sentence, 'tool call blocked: the chat tool kill switch is on', keeping every constant name and import path unchanged per the cross-lane constraint. Added test_kill_switch_refusal_wording_is_unified_everywhere in Tests/Chat/test_console_agent_bridge.py asserting all four constants plus BuiltinToolGate.check()'s actual return value are equal (not importing builtin_tool_gate at module scope in console_agent_bridge.py, preserving its lazy-import boundary), and updated two pre-existing literal pins (Tests/Agents/test_denial_anti_retry_copy.py, Tests/Chat/test_console_activity_presentation.py) that asserted the old wording verbatim. Verified via baseline-vs-fixed comparison (git stash) that all touched suites pass except two pre-existing, unrelated failures already present on HEAD before this change (a resumed-marker redaction test in test_console_agent_bridge.py and 20 filesystem-ledger/scratch-space tests in test_local_tool_provider.py plus 2 catalog-composition tests in test_mcp_tool_provider.py, all reproducing identically with this task's changes stashed out); ran ./scripts/preflight.sh clean (no CSS bundle touched, since the legend/hint Static CSS already had height:auto). Files changed: tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py, tldw_chatbook/Chat/console_chat_controller.py, tldw_chatbook/Agents/mcp_tool_provider.py, tldw_chatbook/Agents/local_tool_provider.py, tldw_chatbook/Chat/console_agent_bridge.py, tldw_chatbook/Agents/builtin_tool_gate.py, and five test files (Tests/UI/test_mcp_permissions_mode.py, Tests/UI/test_mcp_workbench.py, Tests/Chat/test_console_agent_bridge.py, Tests/Agents/test_denial_anti_retry_copy.py, Tests/Chat/test_console_activity_presentation.py).
<!-- SECTION:NOTES:END -->
