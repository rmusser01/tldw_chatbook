---
id: TASK-32579
title: >-
  Library Media: the Import picker inherits Notes' new arrival focus and focus
  cue, unwalked
status: To Do
assignee: []
created_date: '2026-09-14 22:46'
labels:
  - library
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Flagged by wave-4 group 7 (task-32540). Import once's picker now opens with its File name field focused and shows heavy left and right bars on the focused one of Open / Select folder / Cancel, so Enter's target is visible without colour. Library ▸ Media's Import picker goes through the SAME offer_select_folder seam, so it inherits both changes — but it was never driven, so nobody has seen what they look like there, and Media's picker has its own labels and its own surrounding copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Library ▸ Media's Import picker is driven live at 235x52 and 100x30 and its arrival focus and focus cue are captured
- [ ] #2 Anything the inherited behaviour breaks on that surface is fixed or recorded
- [ ] #3 Docs/User_Guide/library/import-and-export.md states the picker's arrival focus, as notes.md now does for Import once
<!-- AC:END -->
