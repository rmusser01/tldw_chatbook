---
id: TASK-2065
title: 'MCP: inspector empty state teaches and does not clip (F-054)'
status: In Progress
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-03 17:48'
labels:
  - ux-review
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Inspector shows 'Select an item to inspect.' in dead space; at 100x30 it clips mid-word (ds-status-badge fixed height 1). No guidance, no preselection. Evidence: mcp_inspector.py:731. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Empty copy is contextual (what inspection offers),Text no longer clips at 100x30,The single problem row is pre-selected on load when exactly one exists,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (UI copy/CSS/selection-default changes). Steps: 1. RED tests: inspector empty copy is contextual and wraps (region height > 1) at 30-col width; workbench pre-selects the single problem row on first load (and not when zero/multiple problems); update tests pinning the old copy (test_mcp_inspector.py, test_destination_shells.py). 2. mcp_inspector.py: module constant for the new empty copy used by compose() and update_readiness(None); DEFAULT_CSS override #mcp-inspector-state { height: auto; min-height: 1; } (ID specificity beats .ds-status-badge height:1; other consumers untouched). 3. mcp_workbench.py: one-shot _preselect_single_problem_on_load() in reload() after _collect_snapshots (excludes off/opt-in built-in, mirrors the callout path's problem definition; restored view state still wins as explicit user state). 4. Run inspector/workbench/destination-shells/parity tests + ruff.
<!-- SECTION:PLAN:END -->
