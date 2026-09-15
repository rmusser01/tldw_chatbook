---
id: TASK-32652
title: >-
  Library Media Import's Browse picker inherits offer_select_folder and has never been live-walked
status: To Do
assignee: []
created_date: '2026-09-15 17:05'
labels:
  - library
  - media
  - picker
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider carried forward through task-32540 and task-32606. Library ▸ Media ▸
Import pushes `FileOpen(offer_select_folder=True)`
(`UI/Library_Modules/library_ingest_controller.py:2006`), so it inherits both
wave-4's focus-on-mount fix and task-32606's screen-docked footer, but no
live keyboard walk of that door has ever been done -- both waves walked the
Notes doors only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Media Import Browse picker is walked keyboard-only at 235x52 and 100x30
- [ ] #2 Any defect the walk finds is filed or fixed
<!-- AC:END -->
