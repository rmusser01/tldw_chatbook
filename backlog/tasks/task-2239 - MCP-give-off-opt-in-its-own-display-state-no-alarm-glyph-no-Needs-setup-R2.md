---
id: TASK-2239
title: >-
  MCP: give off/opt-in its own display state (no alarm glyph, no 'Needs setup')
  (R2)
status: Done
assignee: []
created_date: '2026-08-04 16:18'
updated_date: '2026-08-04 19:30'
labels:
  - ux-review
  - mcp
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fresh-install status self-contradicts: banner 'off' with ready glyph, table row 'Needs setup', callout 'turned off — Enable'. Post-fix re-review P1. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Off/opt-in has its own muted display state in STATE_LABELS/badge path,Table row no longer reads 'Needs setup' for the off built-in,Off summary is not prefixed with the ready/alarm glyph,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no — UI/copy display-state fix, no schema/storage/contract change (existing readiness model extended with one display-only state; amends nothing in backlog/decisions).

1. readiness.py: add ReadinessState.OFF_OPT_IN (off_opt_in) with muted glyph (small ring), STATE_LABELS 'Off (opt-in)', STATE_CSS_CLASSES 'mcp-status-muted'; builtin_readiness(enabled=False) yields that state (reasons stay NOT_CONFIGURED so is_off_opt_in fallback + detail['enabled'] contract are unchanged).
2. readiness.py worst_state(): non-empty snapshot list whose only rows are off/opt-in resolves to OFF_OPT_IN (muted neutral glyph on the banner) instead of READY; empty list stays READY.
3. CSS: add .mcp-status-muted to components/_agentic_terminal.tcss and rebuild tldw_cli_modular.tcss via build_css.py (check_bundle_sync.py must pass).
4. Update stale comments referencing 'enabled=False -> NEEDS_SETUP' (mcp_servers_mode.py, readiness.py).
5. Tests: readiness model/derivation tests for the new state + worst_state tail; servers-mode pristine-summary + badge/row tests; verify no 'Needs setup' for the off built-in row.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
OFF_OPT_IN display state added (muted glyph+label); worst_state tail resolves all-off aggregate to muted instead of ready glyph; CSS muted class added + bundle rebuilt; tests updated (see task file Implementation Notes).
<!-- SECTION:NOTES:END -->
