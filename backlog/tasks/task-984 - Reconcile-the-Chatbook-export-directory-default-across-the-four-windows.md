---
id: TASK-984
title: Reconcile the Chatbook export directory default across the four windows
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 19:33'
updated_date: '2026-07-27 21:02'
labels:
  - chatbooks
  - config
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three live Chatbook files default the visible export directory to ~/Documents/Chatbooks while ChatbookCreationWindow.py uses get_private_chatbooks_dir(). Found completing TASK-967 and deliberately not acted on: reconciling in either direction risks orphaning exports a user already has on disk, so the decision needs an owner rather than a sweep. Whichever default wins, the other location needs either a migration or a documented statement that pre-existing exports stay where they are.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The export directory default is the same in all four Chatbook windows,It is decided and written down whether pre-existing exports are migrated or deliberately left,No file composes the path by literal where an accessor exists,A test derives the expected default the way the app does
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Grep the whole tree (not just the task's named files) for the ~/Documents/Chatbooks literal to confirm the full set of live windows involved.
2. Read task-967's Implementation Notes for the prior owner's context on why this was deliberately deferred.
3. For each of the three live files (ChatbookExportManagementWindow.py, Chatbooks_Window_Improved.py, Wizards/ChatbookCreationWizard.py), verify there is no existing user-facing override (config key or directory picker) for the export path before swapping the literal for get_private_chatbooks_dir().
4. Replace each Path.home()/"Documents"/"Chatbooks" default with get_private_chatbooks_dir(), removing now-unused imports (secure_private_directory, Path where applicable).
5. Confirm each window still surfaces the resolved directory in its existing UI (status bar / preview Static) without inventing a new affordance.
6. Add regression tests deriving the expected path via the real accessor (get_private_chatbooks_dir()) rather than a literal: default resolution per window/wizard, a data_dir config override relocating the default, and a non-destructive check that pre-existing exports at the old ~/Documents/Chatbooks location are left untouched.
7. Add a parametrized source-scan test guarding against the literal reappearing in any of the four files.
8. Run Tests/Chatbooks/ and the UI tests covering the three windows.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Standardized all four Chatbook windows on get_private_chatbooks_dir() (tldw_chatbook/Chatbooks/database_paths.py), the app's private, hardened per-user data directory, matching the project owner's decision already recorded in the task body. This is a default-only change: no migration code was added or needed, and existing exports at ~/Documents/Chatbooks are left exactly where they are (verified by a dedicated regression test).

Found four windows, not three: ChatbookCreationWindow.py already used the accessor (per task-967). The three that still hardcoded Path.home()/"Documents"/"Chatbooks" were ChatbookExportManagementWindow.py, Chatbooks_Window_Improved.py, and Wizards/ChatbookCreationWizard.py -- exactly the set task-967's notes flagged and deliberately left alone. All three already called secure_private_directory/secure_chatbook_directory (create=True, application_owned=True) on the old literal, so no bare mkdir needed hardening; the fix is a one-line swap of the input path plus import cleanup (dropped the now-unused secure_private_directory import in ChatbookExportManagementWindow.py and the now-unused Path import in Chatbooks_Window_Improved.py).

None of the three files has its own export-directory config key or a working directory picker today (ChatbookCreationWizard.py's "Change Location" button on the preview step has zero handler wired to it -- pre-existing dead UI, out of scope for this task and not touched). The only lever a user has to relocate the default is the general [paths] data_dir config setting, which get_private_chatbooks_dir() already resolves through get_user_data_dir() -- a dedicated test proves that override still wins. Each window already displays the resolved directory through its existing affordance (ChatbookExportManagementWindow's status bar "Storage: {dir}", the wizard preview step's #export-path Static), so no new UI was invented.

Tests added: Tests/Chatbooks/test_chatbook_export_directory_default.py (default resolution per window/wizard step, config-override relocation, and a non-destructive regression test that plants a marker file at the old ~/Documents/Chatbooks location and asserts it is untouched after construction). Extended Tests/Chatbooks/test_chatbook_database_paths.py with a parametrized source-scan guarding all four files against the literal reappearing. Every expected path in these tests is computed by calling get_private_chatbooks_dir() itself (with get_user_data_dir monkeypatched), never re-spelled as a literal.

Verified: Tests/Chatbooks/ (169 passed, 1 skipped), plus Tests/UI/test_chatbook_action_recovery_tooltips.py, test_chatbook_management_server_jobs.py, test_chatbooks_screen_server_actions.py, test_file_picker_action_tooltips.py, test_server_chatbook_service_lease.py (27 passed). ruff/pyflakes clean on all touched files.

Files: tldw_chatbook/UI/ChatbookExportManagementWindow.py, tldw_chatbook/UI/Chatbooks_Window_Improved.py, tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py, Tests/Chatbooks/test_chatbook_export_directory_default.py (new), Tests/Chatbooks/test_chatbook_database_paths.py.
<!-- SECTION:NOTES:END -->
