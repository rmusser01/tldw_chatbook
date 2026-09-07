---
id: TASK-2068
title: 'MCP: small-terminal collapse strategy below ~120 cols (F-057)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-03 21:32'
labels:
  - ux-review
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 100x30 the servers table loses Tools/Auth columns with no scroll affordance, summary clips, rail rows truncate. No collapse strategy. Evidence: mcp-100x30.png. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All primary content and actions remain reachable at 100x30
- [x] #2 No mid-word clipping at 100x30
- [x] #3 Rendered-layout test at 100x30
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (TCSS layout/copy behavior; no widget contract change). Steps: 1. Reproduce at 100x30 in a rendered test: identify what clips (servers table columns, summary line, rail rows). 2. RED rendered-layout test at 100x30 asserting primary content/actions reachable + no mid-word clipping. 3. Lightest TCSS fix consistent with existing patterns (candidate: narrow rail min-widths at small widths, wrap the summary, hide lowest-priority table columns) via existing responsive idioms in the bundle. 4. Run MCP layout/parity/workbench tests + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach (3 layers, all in the DEFAULT_CSS <-> _agentic_terminal.tcss lockstep pattern; bundle regenerated via build_css.py and check_bundle_sync green): (1) Wrap not clip: #mcp-inspector-state + #mcp-overview-summary get width: 1fr + height: auto, and #mcp-overview-summary-row becomes height: auto (kept at 1 for #mcp-detail-title). This also repaired the F-054 override, which held only in bare harnesses: in Textual 8.2.7 app-tier CSS beats widget DEFAULT_CSS on ties, so the bundle carries its own copies, and width: 1fr turned out to be required for wrapping to engage at all (the .ds-status-badge width: auto sized to content and never wrapped). (2) Compact triad: _COMPACT_WIDTH=120; MCPWorkbench toggles .mcp-compact on #mcp-hub-grid in on_resize -> 2fr/7fr/2fr with min-widths 16/30/20; MCPServersMode._fit_columns drops lowest-priority columns (Auth, then Connection) by data-derived width estimation (longest content string + 2/cell padding, calibrated against the DataTable's measured virtual width) and on_resize refits when the table width changes. (3) MCPRail truncation budget is now min(_MAX_ROW_LABEL, width - _ROW_CHROME) with on_resize recompose only on budget CHANGE; this exposed a latent mount-echo race in the rail-level scope-guard slots across recompose generations (one leaked bogus ScopeChanged at mount), so both scope selects moved to the proven per-instance _mcp_mount_echo_value pattern (same as the source select). Files: tldw_chatbook/UI/MCP_Modules/{mcp_servers_mode,mcp_rail,mcp_workbench,mcp_inspector}.py, tldw_chatbook/css/components/_agentic_terminal.tcss, tldw_chatbook/css/tldw_cli_modular.tcss (generated), Tests/UI/test_mcp_workbench.py (new bundled-CSS WorkbenchAppWithBundledCSS + 100x30 rendered test). TDD: the 100x30 test was RED in stages (summary clip, then columns, then geometry) as each layer landed. Verification: 21 passed test_mcp_rail.py; 199 passed test_mcp_workbench.py; 217 passed (rail+servers_mode+inspector+profile_form+server_mutations+phase6); 202 passed (servers/inspector/forms + footer context); 220 passed + 1 skip (visual parity + destination shells incl. 140x42 geometry proofs that compact does not engage at wide widths); ruff clean. ADR: not required (TCSS/layout + local guard-pattern migration, no widget contract change). Not done: no horizontal-scrollbar affordance work (compact columns make it unnecessary at 100x30; the native DataTable scroll remains for extreme narrowness); inspector pane content other than the state line is not width-audited at 100x30; commit cafe5933b.
<!-- SECTION:NOTES:END -->
