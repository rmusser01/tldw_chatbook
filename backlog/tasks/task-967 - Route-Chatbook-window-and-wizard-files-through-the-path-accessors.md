---
id: TASK-967
title: Route Chatbook window and wizard files through the path accessors
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 18:06'
updated_date: '2026-07-27 19:28'
labels:
  - config
  - chatbooks
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While completing TASK-865's sweep, the Chatbook window and wizard files were found bypassing the config path accessors and composing user-data and config paths directly. This is the same drift class the audit exists to close: a literal that is correct today and silently wrong the moment the app resolves that path differently. Deliberately left out of TASK-865's scope so the sweep could land, and recorded here rather than lost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chatbook window and wizard files derive their paths from the accessors,No hardcoded ~/.config/tldw_cli or ~/.local/share/tldw_cli literal remains in those files,A test derives its expected path the same way the app does rather than re-spelling a literal
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Enumerate every Chatbook window/wizard file (ChatbookCreationWindow.py, ChatbookExportManagementWindow.py, ChatbookTemplatesWindow.py, Chatbooks_Window_Improved.py, Chatbooks_Window.py, Wizards/ChatbookCreationWizard.py, Wizards/ChatbookImportWizard.py, Wizards/BaseWizard.py, Screens/chatbooks_screen.py, server_chatbook_service_lease.py) for direct ~/.config/tldw_cli or ~/.local/share/tldw_cli composition.
2. Check task-865's Implementation Notes for the exact deferred finding (the ad-hoc db_paths dict + output_dir literal) and verify current state against it.
3. Fix any remaining bare-mkdir/un-hardened directory creation using secure_private_directory, following the chatbook_importer.py pattern.
4. Clean up any test fixtures that re-spell a literal the production code no longer uses.
5. Run Tests/Chatbooks/ and the relevant Tests/UI/test_chatbook*.py files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The specific defect this task was filed for was already fixed on dev before this task started. Task-865's Implementation Notes named the exact deferred finding: ChatbookCreationWindow.py, ChatbookExportManagementWindow.py, ChatbookCreationWizard.py and ChatbookImportWizard.py each built an ad-hoc db_paths dict from self.app.config_data.get('database', {}) with hardcoded fallback literals like '~/.local/share/tldw_cli/tldw_prompts_db.db', and ChatbookCreationWindow.py's output_dir was a bare Path.home()/'.local'/'share'/'tldw_cli'/'chatbooks' with a bare mkdir. Verified all four files now call the real accessors (get_chatbook_database_paths(), get_private_chatbooks_dir()) with zero literal '~/.config/tldw_cli' or '~/.local/share/tldw_cli' remaining, confirmed by the existing Tests/Chatbooks/test_chatbook_database_paths.py (7/7 passing, including a parametrized source-text check across all four files) and Tests/UI/test_tools_settings_window.py's accessor-parity tests.

A distinct, narrower literal survives in three live files -- ChatbookExportManagementWindow.py, Chatbooks_Window_Improved.py and Wizards/ChatbookCreationWizard.py all default the visible chatbooks export/scan directory to Path.home()/'Documents'/'Chatbooks' (already correctly hardened via secure_private_directory/secure_chatbook_directory in all three, just not derived from get_private_chatbooks_dir()). This is NOT the AC's named literal class and switching it to get_private_chatbooks_dir() (~/.local/share/tldw_cli/<user>/chatbooks, already the location ChatbookCreationWindow.py + Tools_Settings_Window.py use for the modal-based creation flow) would silently stop the management window from finding chatbooks a user already exported via the wizard flow -- exactly the live-data-relocation risk this task's constraints say to stop and report on rather than act on. Reporting it here as a real cross-window inconsistency (two live UI flows disagree on where local chatbook exports live) worth its own follow-up task, not silently reconciled by this one.

Fixed the one remaining no-risk item: Chatbooks_Window.py (dead code -- not imported anywhere in the live app except a `.skip`'d integration test) still built its export path from raw config.get() + bare .mkdir(). Routed it through Chatbooks/database_paths.secure_chatbook_directory (the same helper Chatbooks_Window_Improved.py already uses), matching the chatbook_importer.py hardening pattern the task pointed at. Also deleted Tests/Chatbooks/conftest.py's unused mock_app_config fixture and MockWizardApp class: dead test scaffolding, referenced by zero tests, that re-spelled the exact stale db-path/export-directory literals the production fix above already replaced -- left in place it would have been misleading debris of the same "test repeats a literal instead of deriving it" shape the task warns about.

Files: tldw_chatbook/UI/Chatbooks_Window.py, Tests/Chatbooks/conftest.py. Verified via Tests/Chatbooks/ (159 passed, 1 skipped) and Tests/UI/test_chatbook_action_recovery_tooltips.py, test_chatbook_management_server_jobs.py, test_chatbooks_screen_server_actions.py, test_file_picker_action_tooltips.py (22 passed).
<!-- SECTION:NOTES:END -->
