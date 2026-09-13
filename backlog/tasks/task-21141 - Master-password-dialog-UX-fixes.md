---
id: TASK-21141
title: Master-password dialog UX fixes
status: Done
assignee:
  - '@claude'
created_date: '2026-08-25 06:14'
updated_date: '2026-08-25 06:52'
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
- [x] #1 Escape closes the dialog without applying changes
- [x] #2 Validation errors never obscure the dialog's action buttons and clear on input change
- [x] #3 Password requirements and the forgotten-password consequence are stated before first submit
- [x] #4 A show-password toggle exists for both fields
- [x] #5 Dialog title uses sentence case and 'Set up' as the verb
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read dialog widget; map fields/buttons/validation\n2. Escape cancels (binding); error panel repositioned so buttons stay visible; clears on input\n3. Requirements + forgotten-password consequence stated up front\n4. Show-password toggle; sentence-case title\n5. Unit/Pilot tests; live tmux check
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
K-3 root cause found empirically: the dialog's error Static wore the app-wide .error-message class; _wizards.tcss's blanket rule (border: round + padding 1 + margin 1) inflated the one-line error to ~7 rows in the real app, pushing Cancel/Submit past the container's max-height clip — functional but invisible. (Widget-level tests could never see it: no app stylesheet. The new real-app Pilot test pins error height <= 2 and button visibility.) Fixes: dialog-scoped .password-dialog-error class; error clears on any input change; Escape binding follows the Cancel path (K-2); setup message states the 8-char minimum and the forgotten-password consequence up front (K-1/K-4, phrasing matches EncryptionSetupDialog's existing truthful copy); titles sentence-cased with 'Set up' as verb (K-5); Show-password checkbox toggling both fields (K-6); inner container is a VerticalScroll with max-height 32 so buttons stay focus-reachable on short terminals. Tests: 4 widget-level + 1 real-app; dialog + live-contract suites 88 passed.

Files: tldw_chatbook/Widgets/password_dialog.py, Tests/Widgets/test_password_dialog_ux.py (new), Tests/UI/test_first_run_wizard_live_contract.py.
<!-- SECTION:NOTES:END -->
