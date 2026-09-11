---
id: TASK-32308
title: >-
  File dialogs: retire or surface the hidden Ctrl+L path bar now that the File
  name box accepts a typed path
status: To Do
assignee: []
created_date: '2026-09-11 00:55'
labels:
  - library
  - ux
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-32229 (critique-9 grammar branch, PR #2580) made the visible File name box accept a pasted absolute path (with ~) and jump the tree. The vendored fspicker still has a second, hidden way to type a path — the Ctrl+L path bar documented only in Third_Party/textual_fspicker/ENHANCEMENTS.md. Two mechanisms for one job; the reviewer suggested retiring the hidden one or surfacing it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One documented way to type a path remains, or the second is advertised in the dialog and the user guide
<!-- AC:END -->
