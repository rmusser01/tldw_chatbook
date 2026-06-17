---
id: TASK-104
title: ChatbookExportManagementWindow passes wrong kwargs to ConfirmationDialog
status: Done
assignee: []
created_date: '2026-06-11 21:31'
updated_date: '2026-06-17 19:12'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ChatbookExportManagementWindow.py:970 passes confirm_text/cancel_text but ConfirmationDialog takes confirm_label/cancel_label; the kwargs fall through to ModalScreen and likely TypeError when the delete path runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Delete confirmation opens without TypeError,Labels render as intended
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: bounded dialog parameter bugfix; no storage, service, security, or long-lived architecture decision.

1. Add a focused UI regression that drives ChatbookExportManagementWindow._delete_selected() with a selected chatbook and captures the pushed ConfirmationDialog.
2. Verify the regression fails on current dev because ConfirmationDialog receives unsupported confirm_text/cancel_text kwargs.
3. Replace the incorrect kwargs with confirm_label/cancel_label only.
4. Rerun the focused regression and related Chatbook UI tests.
5. Update TASK-104 acceptance criteria and implementation notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the Chatbook export delete confirmation to pass `confirm_label` and `cancel_label`, matching `ConfirmationDialog`'s constructor. Added a regression test that exercises `_delete_selected()` with a selected exported Chatbook and verifies the dialog opens with the intended Delete/Cancel labels instead of raising a `TypeError`.

Verification:
- `python -m pytest -q Tests/UI/test_chatbook_action_recovery_tooltips.py::test_chatbook_delete_selected_opens_confirmation_with_delete_labels --tb=short`
- `python -m pytest -q Tests/UI/test_chatbook_action_recovery_tooltips.py Tests/UI/test_chatbook_management_server_jobs.py --tb=short`
- `git diff --check`
<!-- SECTION:NOTES:END -->
