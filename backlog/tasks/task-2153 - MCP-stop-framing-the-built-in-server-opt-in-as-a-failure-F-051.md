---
id: TASK-2153
title: 'MCP: stop framing the built-in server opt-in as a failure (F-051)'
status: Done
assignee: []
created_date: '2026-08-03 16:24'
updated_date: '2026-08-03 17:01'
labels:
  - mcp
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The built-in server ships disabled, so readiness.aggregate_summary() reports '0 of 1 servers ready — 1 needs setup' and files a problem callout on a pristine install. Treat the disabled built-in as an OFF/opt-in state, not a defect: exclude it from needs-setup math (show it separately as off/opt-in) and present an Enable affordance rather than a problem callout for it. Keep genuine problems (missing credentials etc.) flowing through the existing callout path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 aggregate_summary excludes the disabled built-in from needs-setup math and reports it separately as off/opt-in
- [x] #2 The disabled built-in is presented with an Enable affordance instead of a problem callout
- [x] #3 Genuine problems still flow through the existing callout path
- [x] #4 Readiness tests updated and passing plus ruff clean
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (presentation policy for one built-in row; readiness function signatures unchanged, no schema/boundary change). Steps: 1. RED tests: aggregate_summary excludes the off built-in from ready/needs-setup math and reports it separately as off (pristine, mixed, and genuine-problem cases); worst_state ignores the off built-in; servers-mode renders an Enable affordance (no problem callout) for the off built-in whose click posts BuiltinFlagChanged('enabled', True); genuine problems still produce problem callouts. 2. readiness.py: add is_off_opt_in(); partition in aggregate_summary(); skip off-opt-in in worst_state(). 3. mcp_servers_mode.py: exclude off-opt-in from problem callouts; render #mcp-builtin-enable affordance that posts BuiltinFlagChanged('enabled', True) (performs the fix via the existing config-write path). 4. Update Task A's builtin callout test and the phase6 recovery test to the new affordance. 5. Run MCP tests + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: new readiness.is_off_opt_in() identifies the disabled built-in (detail['enabled'] is False, or NOT_CONFIGURED fallback for hand-built snapshots). aggregate_summary() partitions it out of the ready/needs-setup math and appends 'Built-in server is off (opt-in).' (or a standalone calm line when it is the only snapshot); worst_state() skips it so the aggregate glyph stays ready/neutral on a pristine install. Servers mode (mcp_servers_mode.py update_overview) excludes off-opt-in snapshots from problem callouts and renders a '#mcp-builtin-enable' affordance ('tldw_chatbook (built-in) is turned off — Enable', technical detail in tooltip) whose press posts BuiltinFlagChanged('enabled', True) — the same message the detail Enabled checkbox sends, so the click performs the fix through the existing config-write + resync path. Genuine problems still file mcp-callout-N buttons and count in the summary. Files: tldw_chatbook/MCP/readiness.py, tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py; tests: Tests/MCP/test_readiness_model.py + test_readiness_derivation.py (summary/worst_state), Tests/UI/test_mcp_servers_mode.py (3 new tests; Task A's builtin callout test replaced by the affordance test), Tests/UI/test_product_maturity_phase6_recovery_docs.py (selector #mcp-callout-0 -> #mcp-builtin-enable). TDD: all 5 new tests RED before implementation. Verification: 580 passed (servers_mode, rail, inspector, phase6 recovery, all of Tests/MCP), 313 passed (workbench, visual parity, table-click e2e); ruff clean. ADR: not required (presentation policy; no schema/boundary/contract change). Not done: no workbench-level auto-enable or config default change — the built-in still ships disabled, only the framing changed; commit f0f6d4d49.
<!-- SECTION:NOTES:END -->
