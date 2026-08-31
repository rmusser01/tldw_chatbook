---
id: TASK-25722
title: >-
  Unsaved-changes dialog offers Cancel and Discard changes without
  distinguishing them
status: To Do
assignee: []
created_date: '2026-08-31 05:08'
labels:
  - console
  - ux-review
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dismissing a settings modal with unsaved edits silently adds a third button so the row reads Save, Cancel, Discard changes. Nothing indicates whether Cancel abandons the edits or abandons the dismissal, which is the classic ambiguous-dialog trap. Users must guess which control preserves their work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The dialog offers two unambiguous choices covering discard and continue editing
- [ ] #2 Each button names the outcome it produces rather than a generic verb
- [ ] #3 Controls appearing in response to unsaved edits are announced rather than added silently
<!-- AC:END -->
