---
id: TASK-2065
title: 'MCP: inspector empty state teaches and does not clip (F-054)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-03 18:07'
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
- [x] #1 Empty copy is contextual (what inspection offers)
- [x] #2 Text no longer clips at 100x30
- [x] #3 The single problem row is pre-selected on load when exactly one exists
- [x] #4 Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (UI copy/CSS/selection-default changes). Steps: 1. RED tests: inspector empty copy is contextual and wraps (region height > 1) at 30-col width; workbench pre-selects the single problem row on first load (and not when zero/multiple problems); update tests pinning the old copy (test_mcp_inspector.py, test_destination_shells.py). 2. mcp_inspector.py: module constant for the new empty copy used by compose() and update_readiness(None); DEFAULT_CSS override #mcp-inspector-state { height: auto; min-height: 1; } (ID specificity beats .ds-status-badge height:1; other consumers untouched). 3. mcp_workbench.py: one-shot _preselect_single_problem_on_load() in reload() after _collect_snapshots (excludes off/opt-in built-in, mirrors the callout path's problem definition; restored view state still wins as explicit user state). 4. Run inspector/workbench/destination-shells/parity tests + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: (1) _EMPTY_STATE_COPY constant ('Pick a server, tool, or entry to see what's wrong and what you can do.') shared by compose() and update_readiness(None). (2) MCPInspector.DEFAULT_CSS adds #mcp-inspector-state { height: auto; min-height: 1; } -- ID specificity beats the shared .ds-status-badge height:1; other consumers untouched (verified: full destination shells + parity suites green). (3) MCPWorkbench._preselect_single_problem_on_load() runs in reload() after _collect_snapshots: one-shot per mount (_did_initial_preselect), excludes the off/opt-in built-in via is_off_opt_in (same problem definition as the recovery callouts), and yields to restored view state (explicit user state wins). Files: tldw_chatbook/UI/MCP_Modules/mcp_inspector.py, mcp_workbench.py; tests: test_mcp_inspector.py (new copy + narrow-width wrap tests; old-copy test reworked), test_mcp_workbench.py (3 new preselection tests incl. no-re-hijack; 8 form-flow tests now clear the heuristic selection via _clear_initial_preselection helper), test_destination_shells.py (copy update). TDD: 4 tests RED before implementation. Verification: 198 passed test_mcp_workbench.py; 262 passed + 1 skip (inspector + servers_mode + destination_shells + phase6 + 2 MCP geometry parity tests); ruff clean. ADR: not required (UI copy/CSS/selection default). Not done: pre-selection is Servers-source-agnostic but only fires on first load per mount; the bare InspectorApp harness's 3fr-width quirk (triples a single-child screen width) documented in the wrap test rather than fixed (pre-existing Textual layout behavior); commit bfa1dddea.
<!-- SECTION:NOTES:END -->
