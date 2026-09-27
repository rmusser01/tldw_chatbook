---
id: TASK-33062
title: Theme picker keys are discoverable and help copy matches the controls
status: To Do
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P2. The list's keys (Enter, t, c, n, e, r, Delete, i) work but appear nowhere; F1 says no category shortcuts exist and three places (F1, the ownership boundary copy, the inspector Save row) name an Apply button that no longer exists. Spec §5 promised the keys in the footer. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The footer and F1 help for Theme list the picker's keys
- [ ] #2 No Settings copy refers to an editor Apply button; the copy describes Use/Try and the editor's Save
- [ ] #3 Tests that pinned the stale 'Apply/Save/Reset' copy are updated
<!-- AC:END -->
