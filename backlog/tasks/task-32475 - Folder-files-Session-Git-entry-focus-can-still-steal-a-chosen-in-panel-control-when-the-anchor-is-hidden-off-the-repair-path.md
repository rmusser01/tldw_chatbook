---
id: TASK-32475
title: >-
  Folder files Session Git: entry focus can still steal a chosen in-panel
  control when the anchor is hidden off the repair path
status: To Do
assignee: []
created_date: '2026-09-11 23:18'
labels:
  - library
  - file-notes
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The three-way rule `_entry_focus_is_still_ours` (`tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py` ~3374-3421) returns True when the user has Tabbed to a different still-visible in-panel control and the anchor is then hidden through a path that never goes through `_repair_focus_to`'s repointing -- so `_render_rows` would take focus from the control the user chose. The common real-world path was fixed in task-32265's fix round (identity-anchored entry focus, 10/10); this residual combination has no shipped pin. Rider from the wave-3 scoped re-review of task-32265 / task-32248.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A RED->GREEN test drives that exact sequence on the real panel
- [ ] #2 The rule then defers to any focused in-panel widget that is not the anchor
- [ ] #3 The panel's existing focus pins stay green
<!-- AC:END -->
