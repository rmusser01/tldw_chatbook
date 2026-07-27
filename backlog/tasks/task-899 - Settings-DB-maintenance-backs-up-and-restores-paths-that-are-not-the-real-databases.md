---
id: TASK-899
title: >-
  Settings DB maintenance backs up and restores paths that are not the real
  databases
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 06:00'
updated_date: '2026-07-27 14:40'
labels:
  - settings
  - bug
  - data-safety
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing TASK-860 (the Evals DB ignoring the configured profile). The same class of path bug is in the Settings database-maintenance panel, and there it reaches backup and restore.

`Tools_Settings_Window._get_database_path()` builds its path map from `db_config.get("<name>_db_path", "<hardcoded default>")`. Those hardcoded defaults do not match reality in two independent ways:

1. **Wrong filenames.** It claims `tldw_evals_db.db`, `tldw_prompts_db.db`, `tldw_media_db.db`, `tldw_rag_db.db`. The real files are `evals.db`, `tldw_chatbook_prompts.db`, `tldw_chatbook_media_v2.db`.
2. **No profile directory.** It points at `~/.local/share/tldw_cli/<file>`, but every database actually lives under `~/.local/share/tldw_cli/<profile>/<file>`.

The project already has correct resolvers and this panel does not use them. Verified on this machine:

```
config.get_prompts_db_path()  -> ~/.local/share/tldw_cli/default_user/tldw_chatbook_prompts.db
settings UI hardcodes         -> ~/.local/share/tldw_cli/tldw_prompts_db.db
```

There is no `get_evals_db_path()` helper at all, and neither `evals_db_path` nor `rag_db_path` is defined in `config.py`, so for those two the wrong hardcoded fallback always wins. The file also disagrees with itself: `media_db_path` is read with two different defaults (`tldw_cli_media_v2.db` in one place, `tldw_media_db.db` in another), and `prompts_db_path` likewise.

**Why this is data-safety and not cosmetic.** `_get_database_path()` feeds four workers: `_vacuum_single_worker`, `_backup_single_worker`, `_restore_single_worker` and `_check_single_worker`. So a user who opens Settings and backs up a database is backing up a path that does not exist, and a user who restores is writing a backup to a location that is not the live database — while the real one is never touched. Both operations can report success while doing nothing, which is exactly the wrong failure mode for a feature people reach for during recovery.

The blast radius is not limited to Evals: because none of the paths carry the profile segment, every database is affected for any user whose profile is not the literal default, and `evals`/`rag` are wrong for everyone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The maintenance panel resolves every database through the project's own path resolvers, not hardcoded literals
- [x] #2 Paths honour the configured profile, matching where the databases actually are
- [x] #3 A missing or unresolvable database reports a clear failure instead of silently succeeding
- [x] #4 Backup followed by restore round-trips against the real file, proven by a test
- [x] #5 The duplicated, disagreeing per-key defaults inside the file are gone
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read _get_database_path, the four single-db workers, _update_database_sizes, and _import_chatbook in Tools_Settings_Window.py; read config.py's existing get_*_db_path resolvers; read Evals/eval_orchestrator.py::_initialize_database and RAG_Search/ingestion_indexing.py::_default_indexing_db to find the real Evals and RAG paths.
2. Add get_evals_db_path() and get_rag_indexing_db_path() to config.py, matching the orchestrator/ingestion module exactly; have both delegate to the new helpers (single source of truth).
3. Replace _get_database_path()'s hardcoded path_map with a _DB_PATH_RESOLVERS map to the six config.py resolvers; keep 'rag' (rag_indexing.db is a real, confirmed file) rather than removing it.
4. Make all four single-db workers fail loudly: notify+return on an unresolved path, notify+return on a missing file (vacuum/backup/check), and on restore verify the target directory exists before shutil.copy2 instead of writing unconditionally.
5. Fix self.call_from_thread -> self.app.call_from_thread in all four workers (Widget has no call_from_thread; only App does) so the new error/success notifications actually reach the UI instead of raising AttributeError.
6. Extend _update_database_sizes to cover all six databases via the same resolver map instead of three hardcoded ones.
7. Unify _import_chatbook's duplicate default map to build db_paths from the same resolver map.
8. Write tests: resolver-agreement tests for evals/rag against the orchestrator/ingestion module, a parametrized real backup->restore round-trip test for all six databases, and tests proving unresolvable/missing databases report failure, not silent success. Revert the fix and confirm every new test fails first.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the hardcoded, profile-unaware path_map in Tools_Settings_Window._get_database_path() with a single _DB_PATH_RESOLVERS class map pointing at config.py's real resolvers (get_chachanotes_db_path, get_media_db_path, get_prompts_db_path, get_evals_db_path, get_rag_indexing_db_path, get_subscriptions_db_path). _update_database_sizes and _import_chatbook's separate, disagreeing default map were both rewritten to use the same resolver map -- one source of truth.

Added get_evals_db_path() and get_rag_indexing_db_path() to config.py. eval_orchestrator.py::_initialize_database and ingestion_indexing.py::_default_indexing_db were both changed to call these new helpers instead of inlining the same computation, so app and maintenance panel cannot drift apart again. Ran Tests/Evals/ (436 passed) and the two RAG ingestion test files (66 passed) after this change -- no regressions.

RAG: kept "rag" in the map. rag_indexing.db is a real, confirmed file (found on disk under existing profiles and in RAG_Search/ingestion_indexing.py::_default_indexing_db), so this is not a phantom entry -- it is the RAG indexing-state DB, not a separate "main" RAG database (none exists).

Fail-loudly: all four single-db workers now notify severity="error" and return when _get_database_path() returns None (unresolvable), and notify severity="warning" and return when the resolved file doesn't exist yet (vacuum/backup/check), instead of silently doing nothing. _restore_single_worker now verifies db_path is not None and db_path.parent.exists() before shutil.copy2, erroring out otherwise instead of writing to a location that was never a real database directory.

Found and fixed an additional, related bug while implementing the fail-loudly requirement: all four workers called self.call_from_thread(...), but call_from_thread only exists on Textual's App, not on Widget/Container (ToolsSettingsWindow extends Container) -- confirmed empirically (AttributeError). Every one of these workers' notify calls, including the pre-existing "restored successfully" one, would have raised AttributeError at runtime instead of reaching the user; with exit_on_error=True (the @work default used here) that error propagates and can crash the worker. Fixed by switching to self.app.call_from_thread(...), which is the exact pattern already used correctly by the sibling "all databases" workers 12 lines away in the same file.

Removed redundant local `import shutil`/`import json` inside the affected workers (module already imports both at top).

Tests: added 13 new tests to Tests/UI/test_tools_settings_window.py -- resolver-map coverage, evals/rag resolver-vs-application agreement, a parametrized real backup->restore round trip for all 6 databases (creates a real sqlite file at the resolved path, corrupts it, restores, asserts original content is back at the same path the app uses), and two failure-mode tests (unresolvable db, missing-file db) asserting an error/warning is reported and success is never falsely reported. Verified every new test fails against the pre-fix code: saved the fix as a patch, `git checkout --` the 4 source files back to HEAD, reran -- all 13 failed with the exact expected causes (missing _DB_PATH_RESOLVERS/get_evals_db_path/get_rag_indexing_db_path, stale hardcoded literals still present, and the call_from_thread AttributeError), then reapplied the patch and confirmed all 13 pass again.

Not fixed (out of scope, flagged for a follow-up task): the "vacuum/backup/check ALL databases" workers (_vacuum_worker, _backup_worker, _integrity_worker) and the conversation/character export workers have their own, separate copies of the same hardcoded/profile-unaware literals -- they don't go through _get_database_path() at all. Also, ChatbookImporter expects capitalized keys ("ChaChaNotes", "Prompts", "Media") but _import_chatbook's db_paths dict (both before and after this fix) uses lowercase keys ("chachanotes", "prompts", "media"), so chatbook import likely never received a usable path even before this change -- a pre-existing, separate bug, unchanged by this fix.

Follow-up (same session, post-review): three more bare `self.call_from_thread(...)` calls were found in `_import_chatbook_worker` (lines ~6739/6749/6758) -- the same class of bug, in the worker fed by `_import_chatbook`, the method whose duplicated path map this task already unified. Fixed identically (`self.app.call_from_thread(...)`). Verified `grep -n "self\.call_from_thread" tldw_chatbook/UI/Tools_Settings_Window.py` now returns zero matches (39 calls total, all `self.app.call_from_thread`). Added a regression guard, `test_no_bare_call_from_thread_calls_in_tools_settings_window`, that source-scans the module for the literal `self.call_from_thread(` and asserts `Container`/`ToolsSettingsWindow` have no such attribute while `App` does (documents why the bare form is wrong). Verified the guard fails when the bug is reintroduced (temporarily reverted the 3 calls, reran -- failed with "found 3 bare call(s)" -- then restored and reran -- passed) before trusting it. Re-ran the same test files as before with the same results (6 pre-existing, unrelated `chat_api_key` failures; 30 passed including the new guard test; 16 skipped) plus Tests/Evals/ (436 passed) and the RAG ingestion tests (66 passed) -- no regressions.

Modified files: tldw_chatbook/config.py, tldw_chatbook/UI/Tools_Settings_Window.py, tldw_chatbook/Evals/eval_orchestrator.py, tldw_chatbook/RAG_Search/ingestion_indexing.py, Tests/UI/test_tools_settings_window.py.
<!-- SECTION:NOTES:END -->
