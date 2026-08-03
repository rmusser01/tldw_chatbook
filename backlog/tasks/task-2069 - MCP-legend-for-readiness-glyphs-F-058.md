---
id: TASK-2069
title: 'MCP: legend for readiness glyphs (F-058)'
status: In Progress
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-03 21:37'
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
- [ ] #1 Servers mode exposes a glyph legend (inline or one keypress away),Built-in marker is labeled or legended,Tests/snapshot updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (UI copy addition, mirrors existing legend). Steps: 1. RED test: Servers overview exposes #mcp-servers-legend with all six readiness glyphs + labels and the ⌂ built-in marker. 2. mcp_servers_mode.py: _SERVERS_LEGEND_TEXT derived from STATE_GLYPHS/STATE_LABELS (single source of truth) + '⌂ built-in'; Static after #mcp-overview-callouts; dim CSS rule mirroring #mcp-perm-legend (-muted raw token for bare-harness safety). 3. Run servers_mode/workbench/parity/phase6 tests + ruff.
<!-- SECTION:PLAN:END -->
