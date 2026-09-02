---
id: TASK-28009
title: Library media list - read markers for sequential review
status: To Do
assignee: []
created_date: '2026-09-02 04:10'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nothing marks an item as opened or reviewed; a user reviewing a whole set keeps the which-ones-are-left ledger in their head across pager pages. Persist a lightweight viewed state per item and render it as a row glyph, so a sequential pass over a conference, a tag-filtered set, or a hand-picked collection has visible progress. The reading-scope service (read-it-later flag) is the persistence precedent.

Dev-tip foundations noted 2026-09-02: per-item reading POSITION is already persisted (library_media_reading_progress worker drain, library_screen.py ~42510), and the loaded row carries a "Loaded in Reader" tag - a read/reviewed marker can build on both precedents.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening an item marks it and the list shows the mark after returning
- [ ] #2 Marks survive app restart
- [ ] #3 Unreviewed items are distinguishable at a glance across pages
<!-- AC:END -->
