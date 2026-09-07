---
id: TASK-2069
title: 'MCP: legend for readiness glyphs (F-058)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-03 21:42'
labels:
  - ux-review
  - mcp
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Six readiness glyphs plus the built-in marker have no legend on the Servers canvas (Permissions mode has one). Status is recall, not recognition. Evidence: mcp_rail.py:78. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Servers mode exposes a glyph legend (inline or one keypress away)
- [x] #2 Built-in marker is labeled or legended
- [x] #3 Tests/snapshot updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (UI copy addition, mirrors existing legend). Steps: 1. RED test: Servers overview exposes #mcp-servers-legend with all six readiness glyphs + labels and the ⌂ built-in marker. 2. mcp_servers_mode.py: _SERVERS_LEGEND_TEXT derived from STATE_GLYPHS/STATE_LABELS (single source of truth) + '⌂ built-in'; Static after #mcp-overview-callouts; dim CSS rule mirroring #mcp-perm-legend (-muted raw token for bare-harness safety). 3. Run servers_mode/workbench/parity/phase6 tests + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: _SERVERS_LEGEND_TEXT derived from STATE_GLYPHS/STATE_LABELS ('● ready · ◐ checking · ○ needs setup · ! needs attention · ∅ no tools · ◌ stale') plus '⌂ built-in' -- one source of truth, cannot drift. Rendered as #mcp-servers-legend Static (markup=False) after #mcp-overview-callouts in the Servers overview; dim styling via MCPServersMode.DEFAULT_CSS (-muted raw token, mirroring #mcp-perm-legend's bare-harness rationale and placement). File: tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py; test: new test_overview_shows_readiness_glyph_legend in Tests/UI/test_mcp_servers_mode.py (RED before implementation), STATE_LABELS import added. Verification: 51 passed test_mcp_servers_mode.py; 203 passed (workbench incl. the 100x30 layout test + 2 MCP geometry parity tests + phase6 recovery); ruff clean. ADR: not required (UI copy addition mirroring an existing pattern). Not done: no legend in the rail itself (the overview line covers the same glyphs; a second copy would be clutter); commit b94f1eaa7.
<!-- SECTION:NOTES:END -->
