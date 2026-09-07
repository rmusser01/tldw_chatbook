---
id: TASK-858
title: >-
  Reconcile the evals/prompts/media/rag/subscriptions DB path maps with
  get_*_db_path()
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 04:35'
updated_date: '2026-07-27 16:29'
labels:
  - security
  - db
  - config
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The evals DB alone is named three different, disagreeing ways: Event_Handlers/eval_db_operations.py:28 hardcodes Path.home()/".config"/"tldw_cli"/"evals.db"; the real accessor, Evals/eval_orchestrator.py:90-99, derives get_user_data_dir()/<user_id>/"evals.db"; and UI/Tools_Settings_Window.py:6493 reads a config key evals_db_path that is declared nowhere in config.py's [database] defaults (nor are rag_db_path or subscriptions_db_path). A sandboxed reproduction showed the literal resolving to .../.config/tldw_cli/evals.db, the real accessor resolving to .../.local/share/tldw_cli/default_user/evals.db, and the settings key resolving to '<undeclared>'.

The same defect class appears at UI/Tools_Settings_Window.py:6480-6507 and :6631-6652: the DB path map backing the integrity-check, backup, vacuum, and chatbook-import maintenance operations omits the <user_folder> segment for all six databases it lists, AND uses wrong filenames for at least two of them (tldw_prompts_db.db vs. the real tldw_chatbook_prompts.db; tldw_media_db.db vs. the real tldw_chatbook_media_v2.db). Run as written, these maintenance operations operate on files the app never opens -- an integrity check or vacuum against these paths silently does nothing to the real, live databases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 eval_db_operations.py resolves the evals DB path the same way Evals/eval_orchestrator.py does (via get_user_data_dir(), not a Path.home()/'.config' literal)
- [x] #2 The DB path map behind the Settings screen's integrity check, backup, vacuum, and chatbook-import features is rebuilt from the real get_*_db_path() accessors for every database it lists, with corrected filenames
- [x] #3 evals_db_path, rag_db_path, and subscriptions_db_path are either declared as real config defaults or removed as dead config-key references
- [x] #4 A test asserts each maintenance-operation path equals its corresponding get_*_db_path() return value, not a hand-typed literal, for all six databases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Check current state: TASK-899 (already on origin/dev before this branch) may already have reconciled the Settings maintenance DB-path map and added get_evals_db_path()/get_rag_indexing_db_path()/get_subscriptions_db_path() accessors -- verify before redoing work.
2. Fix Event_Handlers/eval_db_operations.py's remaining Path.home()-based literal to call get_evals_db_path().
3. Declare evals_db_path, rag_indexing_db_path and subscriptions_db_path as real [database] config defaults (they are live, used keys -- not dead references).
4. Add/confirm tests covering all of the above.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC #2 and AC #4 (the Settings screen's DB-path map and its six-database test) were ALREADY DONE on this branch's base -- an earlier, already-merged task (referenced in-code as TASK-899) rebuilt UI/Tools_Settings_Window.py's _DB_PATH_RESOLVERS dict from the real get_*_db_path() accessors (config.py:6823-6830) and Tests/UI/test_tools_settings_window.py already has test_get_database_path_resolves_via_config_resolvers_and_honours_profile plus per-DB parity tests (test_evals_db_path_matches_orchestrator_resolution, test_rag_indexing_db_path_matches_ingestion_module_resolution) and a parametrized backup/restore round-trip test over all six databases. Verified these still pass unchanged. What was genuinely still broken: Event_Handlers/eval_db_operations.py:28 still hardcoded Path.home()/'.config'/'tldw_cli'/'evals.db' -- fixed to call config.get_evals_db_path() (the same accessor Evals/eval_orchestrator.py delegates to), with new regression tests in Tests/Event_Handlers/test_eval_db_operations_path.py (default matches get_evals_db_path(), tracks a retargeted TLDW_CONFIG_PATH profile, explicit db_path unaffected). Also: evals_db_path, rag_indexing_db_path (the task text calls it 'rag_db_path', but the real, only key ever read is rag_indexing_db_path -- 'rag_db_path' does not appear anywhere in the codebase) and subscriptions_db_path were declared nowhere in config.py's [database] TOML template even though their get_*_db_path() accessors are real and already used elsewhere. None were dead (all three accessors have live callers), so declared them as real defaults following the exact sentinel-literal convention the three existing entries (chachanotes/prompts/media) already use, with their correct real fallback filenames (evals.db, rag_indexing.db, tldw_chatbook_subscriptions.db) -- functionally a no-op for existing users (the accessor's custom-path branch only fires when a value differs from this template's sentinel; an unset key already fell through to the correct get_user_data_dir()-based path before this change) but closes the 'declared nowhere' gap and documents the override for new installs. Verified with a sandboxed-HOME probe that the three keys now resolve in DEFAULT_CONFIG_FROM_TOML. Files: tldw_chatbook/Event_Handlers/eval_db_operations.py, tldw_chatbook/config.py, Tests/Event_Handlers/test_eval_db_operations_path.py (new).
<!-- SECTION:NOTES:END -->
