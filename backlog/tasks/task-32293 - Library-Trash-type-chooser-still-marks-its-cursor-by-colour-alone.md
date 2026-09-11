---
id: TASK-32293
title: >-
  Library Trash: the type chooser still marks its cursor by colour alone
status: To Do
assignee: []
created_date: '2026-09-10 12:40'
labels:
  - library
  - media
  - accessibility
  - critique-9
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32210 gave the Media list's type and sort choosers the house `█` cursor, because a 1.09:1 background swap is colour-only and invisible in a plain-text capture. The Trash canvas's own type chooser (`library_media_trash_canvas.py`, `#library-media-trash-type-choices`) was deliberately out of that task's scope and still uses a plain `OptionList`, so two choosers a user reaches from the same toolbar now mark their cursor differently. Evidence: task-1 review of the critique-9 media-list wave, finding 8.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] The Trash type chooser marks its cursor with the same `█` bar as the Media list choosers, with `✓` still marking the active value
- [ ] A painted-cell test pins the bar on the focused row and its absence on the others
<!-- AC:END -->
