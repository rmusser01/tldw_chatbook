---
id: TASK-2243
title: 'MCP: rail status word-badges at width / legend placement (R2)'
status: Done
assignee: []
created_date: '2026-08-04 16:18'
updated_date: '2026-08-04 19:54'
labels:
  - ux-review
  - mcp
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rail glyph-only status needs the dim bottom legend to decode; legend wraps/truncates at 100 cols. Post-fix re-review P3. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Status is decodable without the legend at normal widths (word-badge or state word in row),Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no — rail presentation change only.

Choice (task says pick the lightest option): inline compact legend under the rail's 'Servers' heading, shown when rows exist, listing ONLY the states present among current rows + the ⌂ built-in marker — derived from STATE_GLYPHS/STATE_LABELS so it cannot drift. Word-badge-in-row was rejected: at the rail's real widths (~24-46 cols) a <=16-char state-word column re-truncates the built-in label A4 widened the budget to fit, and it would fight the F-057 width-aware truncation machinery.

1. mcp_rail.py: _present_states_legend() helper + Static #mcp-rail-state-legend composed between the 'Servers' heading and the rows when snapshots exist; DEFAULT_CSS dim rule mirroring #mcp-rail-empty.
2. Tests: present-states-only content (incl. fresh-install 'off (opt-in)' + built-in marker), absent states excluded, no legend at zero rows.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Inline present-states legend under the rail 'Servers' heading (STATE_GLYPHS/STATE_LABELS-derived, + built-in marker); word-badge-in-row rejected on measured rail widths vs A4/F-057 truncation budget; 3 new tests, 272-test sweep green.
<!-- SECTION:NOTES:END -->
