---
id: TASK-32456
title: 'MCP Hub UX Wave E: narrow-width triad'
status: Done
assignee: []
created_date: '2026-09-11 20:57'
updated_date: '2026-09-11 21:10'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ADR-148 Wave E: below 120 columns, stack the inspector under the rail+canvas main row as a bounded scrolling band (max-height 12) instead of squeezing three columns; Advanced auto-collapses while compact. Plan: Docs/superpowers/plans/2026-09-11-mcp-hub-narrow-width-triad.md; spec: Docs/superpowers/specs/2026-09-11-mcp-hub-narrow-width-triad-design.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Wide layouts >=120 cols are geometrically unchanged,Compact layout stacks inspector below the main row as a <=12-row full-width scrollable band,Advanced collapsible auto-collapses at compact without touching the persisted preference,Select prompts render without mid-word breaks at 100x30,All four modes render usably at 100x30
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Executed inline per Docs/superpowers/plans/2026-09-11-mcp-hub-narrow-width-triad.md (3 tasks, TDD). Branch `feat/mcp-hub-ux-wave-d` (Wave E stacked on Wave D — same PR series).
- Layout: `#mcp-hub-main-row` (Horizontal: rail+canvas, ids unchanged) + inspector sibling under `#mcp-hub-grid`; main-row `width: 8fr` preserves the wide 8:3 share exactly. `.mcp-compact` flips the grid to `layout: vertical` (in-repo override precedent) with the inspector as a full-width `max-height: 12; overflow-y: auto` band. CSS landed lockstep in BUNDLED_CSS + widget_defaults_scoped.tcss; the old compact-inspector squeeze rule was deleted in both.
- Advanced auto-collapse: `MCPInspector.apply_compact_layout(compact)` collapses an open `#mcp-adv-collapsible` (widget state only — `_advanced_visible` untouched), driven from `_sync_compact_class`.
- Test adaptations forced by the WIDER canvas at 100 cols (stacking gives the canvas more room — an improvement): the summary-wrap assertion now pins "never clipped mid-sentence" instead of "wraps >=2 lines"; the F-057 column-set pin became "Name/Status never drop + zero horizontal scroll". The intact-prompt test expands the auto-collapsed Advanced first (a user can), then asserts "Overview" renders whole — the old "Overvi|ew" clip is gone.
- Spec open questions resolved to defaults (recorded in plan): band cap fixed at 12; compact rail keeps current sizing (Source select unchanged).
- ADR: covered by ADR-148 decision 2 (linked).
- Verification: test_mcp_workbench 346 passed (full); inspector/rail/servers/permissions + destination MCP subsets 456 passed with ONE flaky destination view-state test (fails under suite load, passes twice in isolation and in the re-run subset — same known flakiness class documented in TASK-32454); doc-contract unchanged at the pre-existing 39 failures.
<!-- SECTION:NOTES:END -->
