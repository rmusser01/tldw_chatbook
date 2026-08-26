---
id: TASK-22301
title: Bound Library rail width across modes
status: To Do
assignee: []
created_date: '2026-08-26 03:31'
labels:
  - library
  - ux
  - layout
dependencies: []
priority: high
references:
  - Docs/superpowers/specs/2026-08-25-library-rail-bounded-width-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the persistent Library navigation rail visually stable across every Library mode by retaining fractional sizing while bounding it around the approved Collections reference width.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With custom widths disabled, every expanded Library rail uses the shared 3:13 default projection and renders within an exact 24–34-cell range.
- [ ] #2 Switching among Media, Chats, Notes, Prompts, Skills, Collections, Search / RAG, Import, and Export at the same settled width keeps the expanded rail edge stable within one compositor cell.
- [ ] #3 The existing explicit custom-width preference remains valid from 24 through 48 cells, applies across ordinary and adaptive Library destinations, and restores unchanged after responsive collapse.
- [ ] #4 Adaptive auto-collapse, five-cell grips, focus recovery, priority, hysteresis, ordinary manual collapse, and compact rail-only/canvas-only takeovers retain their existing behavior without blank reserved rail space.
- [ ] #5 At supported widths, the rail, canvas, adaptive panes, and footer remain contained without intersection; extreme-width escape behavior may hide the rail or compress an explicitly prioritized adaptive rail below 24 rather than overflow.
- [ ] #6 Production-styled tests cover pure projection, initial mount, mode switching, scoped recompose, live resize, custom widths, and applicable 235-, 170-, 120-, 100-, 80-, and 60-column geometry states.
- [ ] #7 Library Settings copy, defaults, ADR-086, the adaptive-reader design, and user documentation describe bounded fractional defaults separately from explicit 24–48-cell custom widths.
<!-- AC:END -->
