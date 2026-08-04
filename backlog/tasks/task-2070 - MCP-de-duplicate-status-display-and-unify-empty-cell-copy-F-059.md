---
id: TASK-2070
title: 'MCP: de-duplicate status display and unify empty-cell copy (F-059)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-03 21:52'
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
- [x] #1 Each status fact is stated once per view
- [x] #2 Empty cells use one consistent placeholder
- [x] #3 Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (copy/aggregation-display change; readiness function signatures unchanged). Scope per task guidance: (a) FULL -- unify empty-cell placeholder to the calmer '—' already used by Tools/Scope/_count_display: _env_auth_display(0) and builtin_readiness auth_display switch from 'none' to '—'. (b) LIGHTEST SAFE PIECE -- aggregate_summary drops the per-state breakdown ('— 1 needs setup, 1 stale'), keeping the aggregate ready count + off/opt-in note; per-server states stay itemized in the table/rail/callouts (the complete list), so no information is lost. Defer (documented): affordance-vs-summary overlap for the off built-in (the summary must remain a coherent sentence on its own) and any deeper layout dedup (redesign). Steps: 1. RED tests: auth '—' at both producers; summary has no per-state breakdown but keeps 'N of M servers ready' and the off note. 2. readiness.py edits. 3. Update affected tests (readiness derivation/model, servers_mode genuine-problem test). 4. Run readiness + servers_mode + workbench + parity + phase6 + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach per the task's conservative scoping: (a) FULL -- empty-cell placeholder unified on the calm '—' already used by the Tools/Scope columns and _count_display: _env_auth_display(0) returns '—' (was 'none') and builtin_readiness sets auth_display='—'; server-source producers already used real values or '—'. (b) LIGHTEST SAFE PIECE -- aggregate_summary() drops the per-state breakdown entirely ('2 of 4 servers ready.' now; off/opt-in note and the calm only-builtin line kept). No information is lost: per-server states remain itemized in the overview table, rail rows, and recovery callouts (the complete list); the summary keeps the aggregate ready count, which the list does not say. Counter import removed (now unused). Files: tldw_chatbook/MCP/readiness.py; tests: Tests/MCP/test_readiness_derivation.py (auth '—' x3), Tests/MCP/test_readiness_model.py (breakdown gone, counts kept), Tests/UI/test_mcp_servers_mode.py (genuine-problem summary assertion). TDD: 6 tests RED before implementation. Verification: 585 passed (Tests/MCP + servers_mode + inspector + rail); 203 passed (workbench + phase6 + 2 MCP geometry parity); ruff clean. ADR: not required (copy/aggregation-display change; function signatures unchanged). DEFERRALS (deliberate, per task guidance): the off-built-in summary note vs the Enable affordance still overlap (the summary must remain a coherent standalone sentence -- removing the note would leave it content-free in mixed views); the deeper 'one fact 3-4 places' layout dedup (summary/table/callout/rail for a single server) is a redesign and was not attempted; the detail pane's prose 'none' (_named_items_text) left as-is since it is a prose line, not a table cell; commit ce43d8433.
<!-- SECTION:NOTES:END -->
