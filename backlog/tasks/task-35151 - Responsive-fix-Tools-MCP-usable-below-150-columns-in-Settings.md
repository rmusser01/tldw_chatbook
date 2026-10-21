---
id: TASK-35151
title: 'Responsive fix: Tools & MCP usable below 150 columns in Settings'
status: Done
created_date: 2026-10-16 20:14
assignee:
- '@agent'
updated_date: 2026-10-21 03:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The workbench grid needs roughly 66 columns; Settings' detail pane gives about terminal width minus 41, so below roughly 150 columns the Tools & MCP category overflows and is unusable. The old full-screen MCP destination worked at 80 columns. Make the category usable in Settings at narrow widths (follow-up to task-35150 / ADR-240).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tools & MCP category is usable at 120 columns
- [x] ADR-148 responsive triad honors Settings pane widths
- [x] Handoff/layout test runs at (120,30)
- [ ] #2 Tools & MCP category is usable at 120 columns,ADR-148 responsive triad honors Settings pane widths,Handoff/layout test runs at (120,30)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Measured overflow inventory (headless probe, real SettingsScreen via DestinationHarness + fake-seam PermissionsHubService):
- Terminal 120 -> detail-body 57, grid 56: compact class IS set (workbench.size.width=56<120) but rail=112, canvas=392, inspector=112 (= 2x/7x/2x the grid) -- canvas fully offscreen.
- Terminal 140 -> grid 61: rail=122, canvas=427, inspector=122 (same degeneracy). Terminal 150 -> grid 71: mins floor the fr shares (16/35/20) and everything fits.
- Root cause: Textual 8.x fr width resolution in a Horizontal degenerates when the children's min-widths cannot all be satisfied -- each child then gets fraction x container width (isolated 61-wide repro reproduces 122/427/122). The compact triad mins (rail 16 + canvas 30 + inspector 20 = 66) exceed the Settings detail pane below ~150 cols. ADR-148 decision 2's stacking (inspector band under a rail+canvas row) was never implemented -- F-057 shipped only the share/min rebalance.
Plan:
1. RED: parametrize the handoff test at (120,30) too (Tests/UI/test_mcp_permissions_handoff.py) and add a focused width test in Tests/UI/test_settings_tools_hub.py asserting the triad panes fit the detail pane at (120,30).
2. Wrap rail+canvas in Horizontal#mcp-hub-row in MCPWorkbench.compose (DOM order [row, inspector] keeps the wide-screen visual order); add .mcp-stacked class toggled from _sync_compact_class below _STACKED_WIDTH=66 (the compact mins sum), keyed on the workbench's own width (already pane-relative).
3. CSS (tokens first): add -agentic-mcp-row-share: 8fr and -agentic-mcp-compact-row-share: 9fr to _variables.tcss; in _agentic_terminal.tcss add #mcp-hub-row share/min rules and the stacked block (grid layout: vertical; row full-width 1fr with stacked mins rail 12 + canvas 24; inspector full-width bounded band height 40%, internally scrolling). Lockstep copies in MCPWorkbench.BUNDLED_CSS; rebuild bundle with build_css.py.
4. GREEN: run handoff at (120,30)/(150,30)/(170,48), new width test, test_mcp_workbench.py, test_settings_tools_hub.py, token governance, visual-parity MCP tests.
ADR required: no -- completes ADR-148 decision 2's stacking design below the width where its own compact minimums cannot fit; the >=66-col triad behavior is unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
## Implementation Notes

**Approach.** A headless probe (real SettingsScreen via DestinationHarness + fake-seam PermissionsHubService) measured the actual overflow: at 120/140 columns the Settings detail pane gives the grid 56/61 columns, `.mcp-compact` IS set, but rail/canvas/inspector measure 112/392/112 (= 2x/7x/2x the grid). Root cause: Textual 8.x fr width resolution in a Horizontal degenerates when the children's min-widths cannot all be satisfied -- each child then takes fraction x container width (reproduced in an isolated 61-wide Horizontal). The compact triad minimums (rail 16 + canvas 30 + inspector 20 = 66) exceed the pane below ~150 terminal columns. ADR-148 decision 2's stacked triad was never implemented (F-057 shipped only the share/min rebalance), so below 66 columns there was no working layout.

**Fix.** `MCPWorkbench.compose` now wraps rail+canvas in `Horizontal#mcp-hub-row` (grid children = [row, inspector]; the row carries the rail+canvas fr sum via new single-consumer tokens `$ds-agentic-mcp-row-share: 8fr` / `$ds-agentic-mcp-compact-row-share: 9fr`, so every pane's wide-screen share is unchanged). `_sync_compact_class` additionally toggles `.mcp-stacked` below `_STACKED_WIDTH = 66` (the compact minimums' sum, keyed on the workbench's OWN pane-relative width). `.mcp-stacked` flips the grid to `layout: vertical`: rail+canvas row (full width, stacked canvas min 24) with the bounded, internally-scrolling inspector band (40%, floor 1) below -- ADR-148 decision 2's design. Rules live in `_agentic_terminal.tcss` with bare-harness lockstep copies in `MCPWorkbench.BUNDLED_CSS`; bundle rebuilt via build_css.py. Two collateral content fixes surfaced by the probe/tests: the Tools-mode master-switch Button (auto width ~50) now wraps/caps at the canvas (inspector-action idiom, `MCPToolsMode.BUNDLED_CSS`), and the permissions profile Select gained MCPRail's one-shot per-instance mount-echo guard -- a late-delivered constructor echo was reverting deep-linked profiles to "default" once the wrapper shifted message ordering (latent race, exposed deterministically). The visual-parity forced-loading test was realigned to the stacked contract at 140 columns (the old x-ordering pinned a triad whose canvas actually overflowed offscreen at that width; nothing checked horizontal containment).

**Verification.** RED->GREEN: new `test_tools_triad_fits_settings_detail_pane_at_120_columns` (fails on missing stacked layout at HEAD) and the handoff test re-parametrized to (120,30) (fails with canvas offscreen at HEAD) both green. `test_mcp_permissions_handoff` 6/6 (120/150/170 x 2 themes); `test_mcp_workbench` 349/349; token governance 8/8; settings-tools-hub 11 pass + 5 pre-existing env-blocked (RecoveryRequired, verified identical at HEAD via stash); visual-parity MCP subset matches HEAD (2 pre-existing env-blocked) with forced-loading green under the stacked contract; two batches of MCP layout suites (compact-readability, rail, deferred canvases, permissions/tools modes, inspector access, servers/audit modes, lifetime) show IDENTICAL failure lists at HEAD and with the change (all pre-existing env blocks). Probe re-run: 120 -> rail 16 / canvas 40 / band 56x1 all inside the pane; 150 -> 16/35/20 unchanged from HEAD.

ADR required: no -- completes ADR-148 decision 2's stacking design below the width where its own F-057 compact minimums cannot fit; triad behavior at >=66 columns is unchanged.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
