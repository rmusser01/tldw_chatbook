---
id: TASK-34401
title: 'Shared file picker: directory, bookmark and typed-path text parses as markup'
status: To Do
assignee: []
created_date: '2026-10-04 18:51'
labels:
  - bug
  - crash
  - file-picker
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-34400's sweep of the Roleplay code. Widgets/enhanced_file_picker.py, the import/export picker every screen opens, renders filesystem text through markup-parsing surfaces: breadcrumb buttons and tooltips (directory names along the path), bookmark and recent-location labels, toasts that quote typed paths, bookmark names and OSError text, the inline error line, and the dialog's border title (the caller's title). On Textual 8.2.8 a name containing '[/' (for example a directory named 'a[' followed by '/b]', or one named '[x=y]z') raises MarkupError while drawing, and render-time errors exit the whole app (TASK-32533's keep-alive covers only message handlers). TASK-34400 escaped Roleplay's three picker titles at the call site; if this task makes the border title literal, those escapes must be removed in the same change or a stray backslash shows. The Console staged-handoff strip was also outside TASK-34400's sweep.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A directory, file or bookmark name, or a typed path, containing markup-like text ('[/]', '[b]x', '[x=y]z', '[@click=app.quit]x') never raises and paints literally on every picker surface: breadcrumbs, tooltips, labels, toasts and the error line
- [ ] #2 The picker title paints literally for every caller, and callers that pre-escape their titles (Roleplay's three since TASK-34400) are updated in the same change so no backslash shows
- [ ] #3 Each surface has a regression test that fails on the pre-fix code
<!-- AC:END -->
