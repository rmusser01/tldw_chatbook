---
id: TASK-2070
title: 'MCP: de-duplicate status display and unify empty-cell copy (F-059)'
status: In Progress
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-03 21:43'
labels:
  - ux-review
  - mcp
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
One server state is stated 3-4 times (summary + table row + callout + rail). Auth shows 'none' while Tools shows dash for empty cells. Evidence: mcp_servers_mode.py:562-563. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each status fact is stated once per view,Empty cells use one consistent placeholder,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (copy/aggregation-display change; readiness function signatures unchanged). Scope per task guidance: (a) FULL -- unify empty-cell placeholder to the calmer '—' already used by Tools/Scope/_count_display: _env_auth_display(0) and builtin_readiness auth_display switch from 'none' to '—'. (b) LIGHTEST SAFE PIECE -- aggregate_summary drops the per-state breakdown ('— 1 needs setup, 1 stale'), keeping the aggregate ready count + off/opt-in note; per-server states stay itemized in the table/rail/callouts (the complete list), so no information is lost. Defer (documented): affordance-vs-summary overlap for the off built-in (the summary must remain a coherent sentence on its own) and any deeper layout dedup (redesign). Steps: 1. RED tests: auth '—' at both producers; summary has no per-state breakdown but keeps 'N of M servers ready' and the off note. 2. readiness.py edits. 3. Update affected tests (readiness derivation/model, servers_mode genuine-problem test). 4. Run readiness + servers_mode + workbench + parity + phase6 + ruff.
<!-- SECTION:PLAN:END -->
