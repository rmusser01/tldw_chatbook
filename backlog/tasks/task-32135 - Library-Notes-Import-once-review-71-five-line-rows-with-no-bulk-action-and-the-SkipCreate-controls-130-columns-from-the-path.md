---
id: TASK-32135
title: >-
  Library Notes Import once review: 71 five-line rows with no bulk action and the Skip/Create controls 130 columns from the path
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
  - layout
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both assessors: each review row is five lines ('Ready to import as a new note.' / 'Content: create 1 new note.' / 'Create in vault / Archive.' plus the path and right-aligned Skip/Create); about six rows fit per screen and 25 wheel notches moved from item 1 to item 8. Reviewing 71 items honestly is dozens of identical screens, which undermines the reviewed-mutation principle. The repeat-import review (Unchanged repeat, folder collision) is strong and should keep its grammar. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rows render on one line (path · action · destination) with their controls adjacent to the path
- [ ] #2 Rows are grouped under a header per action class with a per-group Skip
- [ ] #3 At 235x52 at least 15 rows are visible per screen
<!-- AC:END -->
