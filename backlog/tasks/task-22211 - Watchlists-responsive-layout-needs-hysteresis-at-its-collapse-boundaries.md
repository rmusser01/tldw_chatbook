---
id: TASK-22211
title: >-
  Watchlists responsive layout needs hysteresis at its collapse boundaries
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - watchlists
  - ux
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22211).

New with PR #2063. `UI/Watchlists_Modules/region_layout.py:132-175`:
`resolve_effective_layout` applies bare width thresholds with no `previous` state, and
`on_resize` recomputes per Textual Resize event. Crossing 145 columns by ONE cell flips
the right rail: region factory + mount/remove pair per flip
(`watchlists_workbench.py:226-309`), repeated per Resize during a drag. This is the
documented sub-2-cell width-flap trap; the Library media reader carries the fix
(`LAYOUT_HYSTERESIS_WIDTH = 4`, `Library/library_media_reader_state.py:16`, `:341-355`)
and Watchlists does not. Aggravator (medium confidence): `_available_layout_width` prefers
`workbench.size.width` (`watchlists_collections_screen.py:2999`), which is
scrollbar-sensitive — a scrollbar toggle at the boundary could flap the layout with no
user resize.

## Acceptance Criteria

- [ ] Repeated +/-1-cell width changes at a collapse boundary cause no mount/remove churn (hysteresis test at the boundary, both directions)
- [ ] The width source is not flappable by a scrollbar toggle, or a code-level guard absorbs sub-hysteresis changes (the repo rule: never trust a CSS-only guard)
- [ ] Approach consistent with the Library reader's hysteresis precedent
