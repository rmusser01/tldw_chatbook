---
id: TASK-32128
title: >-
  Library Notes Sort control is inert: the tree is title-ordered by a pinned decision and the Title option never renders
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed by the evidence assessor: Newest → Oldest changes the label and nothing else; both modes show strict alphabetical order, which is neither of the DB orders. `test_placement_title_sort_key_matches_repository_tiebreakers` pins title ordering for tree placements, so the control contradicts a decision. 'Title' does not fit the 38-column pane and never renders. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Either the Sort control is removed from the tree view, or Newest/Oldest visibly reorder the rows within each folder
- [ ] #2 The decision is recorded in the task and the pinning test is reconciled with it
- [ ] #3 Every remaining sort option renders at the narrowest pane width
<!-- AC:END -->
