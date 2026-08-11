---
id: TASK-14902
title: Library cycle controls should offer a discoverable option menu
status: To Do
assignee: []
created_date: '2026-08-10 17:20'
labels:
  - library
  - ux
  - recritique-2026-08-09
dependencies: []
priority: low
---

## Description

Filed from task-4023 AC#5. The batch's bounded fix gave every Library value-cycler
its own glyph (`⇄` via `library_cycle_label`) and an option-enumerating tooltip, so
the option set is no longer invisible — but a tooltip is hover/focus-gated and the
control still only ADVANCES; a user cannot jump to a specific option (re-critique
heuristic #6, "cycle-buttons hide their option space"; persona note "cycle-buttons
can't be jumped"). The Notes canvas's Sort control already shows the discoverable
pattern: pressing it swaps in a one-row choice strip with a ✓ on the active option.
Converge the cyclers (media type, prompts sort/collection, skills sort + editor
toggles, export quality, Search/RAG mode) on that choice-strip pattern or a shared
popover, and retire the per-press cycle where it no longer earns its place.

## Acceptance Criteria

- [ ] Every Library cycle control can show its full option set on screen (not only in a tooltip) and lets the user pick an option directly
- [ ] The active option carries a non-colour marker consistent with the Library marker vocabulary
- [ ] The footer/F1 advertise the interaction where it is keyboard-reachable
