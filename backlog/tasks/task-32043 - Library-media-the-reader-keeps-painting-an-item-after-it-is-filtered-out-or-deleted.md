---
id: TASK-32043
title: >-
  Library media: the reader keeps painting an item after it is filtered out or
  deleted
status: To Do
assignee: []
created_date: '2026-09-08 14:36'
labels:
  - library
  - media
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #7 P2. After a filter that yields zero results, and after deleting the item currently loaded in the reader, the reader keeps painting the now-absent item's content instead of reflecting that nothing is selected. The reader should track the visible set.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When the loaded media item leaves the visible set (0-result filter or deletion), the reader clears to its no-selection state (or a 'this item was deleted' note with Undo)
- [ ] #2 Opening or re-filtering to a present item still shows it normally
- [ ] #3 A pin drives a 0-result filter and a delete-of-loaded-item and asserts the reader no longer paints the absent item
<!-- AC:END -->
