---
id: TASK-32385
title: 'Settings still ships the retired "Folder Files" name in the tree-pane controls'
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - settings
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32218 settled one noun per source: the Library's own notes are "Library notes" and the on-disk ones are "Folder files"; "Folder Files" as a user-visible name was retired. The Settings screen still renders the old capitalisation in its tree-pane controls -- `settings_appearance_defaults.py:389` ("Folder Files tree"), `settings_search_index.py:479` and `:483` ("Folder Files tree pane" / "Folder Files tree width") -- plus the matching strings in `settings_screen.py` that key off them. A setting a user searches for by the name the Library taught them should be findable under that name.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No user-visible Settings string reads "Folder Files"; they read "Folder files"
- [ ] #2 The search-index entries and any prefix matching that keys off those strings are updated together, so the settings stay findable
- [ ] #3 A test or the existing copy census pins the retired name out of the Settings surface
<!-- AC:END -->
