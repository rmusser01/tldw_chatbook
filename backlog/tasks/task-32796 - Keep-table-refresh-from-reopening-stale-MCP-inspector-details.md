---
id: TASK-32796
title: Keep table refresh from reopening stale MCP inspector details
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 16:33'
updated_date: '2026-09-18 16:58'
labels:
  - mcp
  - ui
  - selection
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep explicit server navigation and background table refresh from selecting rows on the users behalf, while preserving real keyboard and pointer selection and current inspector actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Changing server clears obsolete tool/finding details and keeps them cleared through redraw without another user gesture.
- [x] #2 Focused table repopulation does not synthesize a selection; real keyboard and pointer selection remain responsive after refresh.
- [x] #3 Deterministic queued-highlight and adjacent shared-table tests cover the actual dispatch boundary without fixed-delay suppression assumptions.
- [x] #4 Targeted checks and native private navigation evidence pass; remaining review and PR evidence are updated.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the two existing failing inspector regressions through the shared table-selection helper and deferred Textual highlights. 2. Reproduce delayed message delivery deterministically and preserve real click/arrow/Enter behavior. 3. Implement ADR-170: change repopulating_table(table) to a synchronous prevent context; update every existing caller plus Audit Findings without spanning awaits. 4. Verify all impacted consumers, real private MCP navigation, independent review and draft PR evidence. ADR required: yes. ADR path: backlog/decisions/170-table-repopulation-selection-boundary.md. Reason: The shared repopulation helper changes its cross-module call contract to make programmatic highlight suppression independent of queue timing; ADR-161 remains the component foundation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stopped table redraws from selecting tools or findings and reopening cleared MCP inspector details. ADR-170 changes the shared helper to a synchronous Textual prevent context at message publication; all callers and Audit Findings are migrated. Gesture dedup includes table identity and resets only the rebuilt table. Tools/Permissions external drills suppress their final programmatic cursor moves while preserving the next real Enter.

Validation: 273 selected consumer cases pass, plus both original inspector-clearing regressions (275 distinct cases). Fourteen new cases cover delayed row/cell delivery, click/arrow/Enter after refresh, hidden-table dedup and focused external drills. Nine delayed-event and three review regressions were observed red before their fixes. Independent review has no remaining actionable finding. Nine contexts contain no awaits; no introduced Ruff findings; changed functions/new files format; diagnostic inventory and Backlog validation pass. No full suite was run.

The initial wider run had four unchanged Audit CSS literal assertions and two speech harness profile-setup failures. The CSS failures were reproduced against the saved baseline; the speech harness cases were not requalified here. Four native dark/light compact/wide journeys and eight inspected captures pass with clean shutdown, ten healthy private databases and unchanged defaults/policies. Native results and the existing compact rail clipping limitation are recorded in Docs/superpowers/qa/2026-09-18-mcp-table-selection/README.md. Scope excludes connected external servers, tool execution, selected-definition/schema currentness and full destination qualification.

Documentation: ADR-170, component-pattern contract, testing lesson, MCP and overall completion ledgers. Production: shared table helper, MCP Tools/Permissions/Servers/Audit and Voice Cloning; tests: test_mcp_table_refresh_selection.py. Draft PR2707 remains subject to its own visual review and merge approval.
<!-- SECTION:NOTES:END -->
