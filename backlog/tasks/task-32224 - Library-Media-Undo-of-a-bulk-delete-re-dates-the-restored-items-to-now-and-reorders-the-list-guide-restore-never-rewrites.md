---
id: TASK-32224
title: >-
  Library Media: Undo of a bulk delete re-dates the restored items to now and
  reorders the list (guide: 'restore never rewrites')
status: To Do
assignee: []
created_date: '2026-09-10 14:55'
labels:
  - library
  - media
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After Undo the restored items read 'now' and jump to the top of Newest; the guide says restore never rewrites the item. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 22.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Restore preserves the item's modified time and position, or the guide states the real behaviour
<!-- AC:END -->
