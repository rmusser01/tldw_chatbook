---
id: task-1625
title: 'Settings: visual-voice pass (phosphor titles, state strip, toast border)'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - settings
  - design
  - critique-r3
dependencies: []
priority: medium
---

## Description (the why)

Both blind panels flagged the same brand gap: interaction design authored, visual voice 'municipal utility'. Owner chose a full visual pass. Constraints: density, text-labeled states, craft-floor (no side-stripe callouts, contrast floors).

## Acceptance Criteria (the what)

- [x] The three pane titles carry the single Focus Phosphor accent
- [x] The State bar reads as a raised status surface (focus steel at 30% alpha)
- [x] Toasts carry a full severity-tinted round border instead of the stock side stripe (DESIGN.md Don'ts compliance)
- [x] RAG 'Profiles' border title readable (was near-invisible)
- [x] Decisions captured in DESIGN.md

## Implementation Notes

All CSS-token work in `_agentic_terminal.tcss` + `_base.tcss` (+ bundle rebuild): `.settings-column-title` color $accent; `.settings-state-banner` background $ds-focus-bg 30%; `Toast` + severity variants full round borders; `.settings-secondary-card` border-title-color/-style. Restrained by design: one accent carrier, one raised strip — density and legibility untouched. Documented under 'Voice carriers' in DESIGN.md.
