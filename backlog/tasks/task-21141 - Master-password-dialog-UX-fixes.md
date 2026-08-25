---
id: TASK-21141
title: Master-password dialog UX fixes
status: To Do
assignee: []
created_date: '2026-08-25 06:14'
labels:
  - ux
  - wizard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings K-1..K-6 (findings.md section K): Escape does not cancel; the min-8-chars error panel covers Cancel/Submit indefinitely; requirements disclosed only after failed submit; no forgotten-password consequence stated; 'Setup Master Password' title casing; no show-password toggle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Escape closes the dialog without applying changes
- [ ] #2 Validation errors never obscure the dialog's action buttons and clear on input change
- [ ] #3 Password requirements and the forgotten-password consequence are stated before first submit
- [ ] #4 A show-password toggle exists for both fields
- [ ] #5 Dialog title uses sentence case and 'Set up' as the verb
<!-- AC:END -->
