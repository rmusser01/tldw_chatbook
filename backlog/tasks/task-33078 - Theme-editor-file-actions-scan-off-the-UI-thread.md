---
id: TASK-33078
title: Theme editor file actions scan off the UI thread
status: To Do
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-32957 (PR #2850 Qodo 4116061870). Each editor file action (Save, Save as, Rename, Delete, Import, Export) still resolves names through a ~210 ms backup-scoped folder scan on the UI thread with 50 theme files, and waits up to one picker scan (~414 ms) when they overlap. File operations must stay in SettingsThemeEditor (the backup participant recognises only that instance). Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With 50 saved themes no editor file action blocks the UI thread for more than 100 ms
- [ ] #2 File operations still run through SettingsThemeEditor's backup scope
<!-- AC:END -->
