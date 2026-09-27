---
id: TASK-33060
title: Discard in the Theme leave prompt restores the pre-editor theme
status: To Do
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P1 (2026-09-27, 28/40). After the editor's Try, choosing Discard on Back, a category switch or any leave prompt leaves an unlisted custom_<name> theme applied app-wide. The picker then marks no theme (or the wrong one) active and Revert cannot undo it. Cause: the editor's Try sets app.theme directly (settings_theme_editor.py on_apply_theme) and no discard path restores it. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Try then Discard (Back, category switch, Escape, quit/navigation prompt), the app runs the theme that was active before the editor opened
- [ ] #2 The picker marks exactly one theme active after a discard, matching app.theme
- [ ] #3 Try then Save keeps the saved theme applied (no regression)
- [ ] #4 A Pilot test pins Try → Discard → original theme for each leave path
<!-- AC:END -->
