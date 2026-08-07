---
id: TASK-2857
title: Unify Library import/export naming across rail, canvases and toasts
status: To Do
assignee: []
created_date: '2026-08-07 01:10'
labels:
  - library
  - ux-copy
  - consistency
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-10; extends the 2026-08-04 critique's naming P1; observed at dev
`6ffa56516`).

One flow, five names: rail button "Add content…" → canvas titled "Import media" → Media empty
state "Ingest something to see it here." → button "Start ingest" → toast "Ingest finished — 1
imported". Siblings use "Import note" (Notes toolbar) and "Import…" (Prompts/Skills). On the
export side: rail "Export" → canvas "Export chatbook" ("chatbook" appears nowhere else in the
UI) → media detail action "Open in Media manager" (a surface never named anywhere else).

First-time users wonder whether these are different features; the naming breaks recognition on
every return visit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One verb pair is chosen (recommend Import/Export) and used consistently across rail rows, canvas titles, empty states, buttons and toasts for the same concept
- [ ] #2 "chatbook" is either introduced with a one-line explainer where it appears or replaced (e.g. "Export bundle (.zip)")
- [ ] #3 "Open in Media manager" names the surface it actually opens using that surface's own name
- [ ] #4 A naming inventory in the task notes lists every changed string (rail/canvas/toast/tooltip), and the user guide pages citing these labels are updated or re-stamped
<!-- AC:END -->
