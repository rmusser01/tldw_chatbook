---
id: TASK-104
title: ChatbookExportManagementWindow passes wrong kwargs to ConfirmationDialog
status: To Do
assignee: []
created_date: '2026-06-11 21:31'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ChatbookExportManagementWindow.py:970 passes confirm_text/cancel_text but ConfirmationDialog takes confirm_label/cancel_label; the kwargs fall through to ModalScreen and likely TypeError when the delete path runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Delete confirmation opens without TypeError,Labels render as intended
<!-- AC:END -->
