---
id: TASK-2062
title: 'MCP: stop framing the built-in server opt-in as a failure (F-051)'
status: In Progress
assignee: []
created_date: '2026-08-03 16:24'
updated_date: '2026-08-03 16:48'
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
- [ ] #1 aggregate_summary excludes the disabled built-in from needs-setup math and reports it separately as off/opt-in,The disabled built-in is presented with an Enable affordance instead of a problem callout,Genuine problems still flow through the existing callout path,Readiness tests updated and passing plus ruff clean
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (presentation policy for one built-in row; readiness function signatures unchanged, no schema/boundary change). Steps: 1. RED tests: aggregate_summary excludes the off built-in from ready/needs-setup math and reports it separately as off (pristine, mixed, and genuine-problem cases); worst_state ignores the off built-in; servers-mode renders an Enable affordance (no problem callout) for the off built-in whose click posts BuiltinFlagChanged('enabled', True); genuine problems still produce problem callouts. 2. readiness.py: add is_off_opt_in(); partition in aggregate_summary(); skip off-opt-in in worst_state(). 3. mcp_servers_mode.py: exclude off-opt-in from problem callouts; render #mcp-builtin-enable affordance that posts BuiltinFlagChanged('enabled', True) (performs the fix via the existing config-write path). 4. Update Task A's builtin callout test and the phase6 recovery test to the new affordance. 5. Run MCP tests + ruff.
<!-- SECTION:PLAN:END -->
