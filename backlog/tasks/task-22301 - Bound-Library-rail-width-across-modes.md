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
- [ ] #1 With custom widths disabled, every Library rail displayed alongside content uses the shared 3:13 default projection and renders within an exact 24–34-cell range; rail-only, hidden/collapsed, and compressed-priority states remain explicit exceptions.
- [ ] #2 Switching among Media, Chats, Notes, Prompts, Skills, Collections, Search / RAG, Import, Export, Study handoffs, and the landing canvas at the same settled width keeps a co-present rail edge stable within one compositor cell.
- [ ] #3 The existing explicit custom-width preference remains valid from 24 through 48 cells, applies to co-present rails across ordinary and adaptive Library destinations, and restores unchanged after responsive collapse or a rail-only compact takeover.
- [ ] #4 Adaptive auto-collapse, five-cell grips, focus recovery, automatic/explicit priority, hysteresis, ordinary manual collapse, and existing Notes-specific takeovers retain their behavior; below 64 columns every ordinary route uses a reversible rail-only/canvas-only emergency stage with no blank reserved space.
- [ ] #5 At supported widths, the rail, canvas, adaptive panes, and footer remain contained without intersection; extreme-width escape behavior may hide panes, compress an explicitly prioritized adaptive pane below its protected minimum, or invoke the approved ordinary `<64` emergency stage rather than overflow.
- [ ] #6 Production-styled tests cover the exact projection oracle and wide/compact box-model inputs, initial/settled mount, every enumerated route, Notes Navigator/work/explicit-priority adaptive branches, scoped recompose, live resize, custom widths, and specified 235-, 170-, 120-, 100-, 80-, and 60-column geometry states.
- [ ] #7 Library Settings copy, defaults, ADR-086, the adaptive-reader design, and user documentation describe bounded fractional defaults separately from explicit 24–48-cell custom widths.
<!-- AC:END -->
