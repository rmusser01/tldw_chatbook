---
id: TASK-32137
title: >-
  Library Notes list rows show no age, duplicate titles are indistinguishable, and long titles clip
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - layout
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PROVEN: tree rows (`_compose_tree_rows`) carry no age label; the two-line title-plus-age code path lives in `_compose_list`, which is dead once a tree projection exists (always, because Agent_Lessons is seeded). Two notes titled 'Reading list' render as identical rows in the tree and in filtered results. The guide says rows show 'title and age'. Depends on task-32127 for width. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tree rows show a relative age
- [ ] #2 Visible duplicate titles get a folder then modified-date suffix at render time
- [ ] #3 The unreachable flat-list age code is used or removed
- [ ] #4 The guide matches
<!-- AC:END -->
