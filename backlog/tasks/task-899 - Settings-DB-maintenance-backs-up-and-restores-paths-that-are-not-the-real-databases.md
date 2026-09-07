---
id: TASK-899
title: >-
  Settings DB maintenance backs up and restores paths that are not the real
  databases
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 06:00'
updated_date: '2026-07-27 15:16'
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

Fail-loudly: all four single-db workers now notify severity="error" and return when _get_database_path() returns None (unresolvable), and notify severity="warning" and return when the resolved file doesn't exist yet (vacuum/backup/check), instead of silently doing nothing.

Found and fixed an additional, related bug while implementing the fail-loudly requirement: all four workers called self.call_from_thread(...), but call_from_thread only exists on Textual's App, not on Widget/Container (ToolsSettingsWindow extends Container) -- confirmed empirically (AttributeError). Fixed by switching to self.app.call_from_thread(...).

Removed redundant local import shutil/import json inside the affected workers (module already imports both at top).

Tests: added 13 new tests to Tests/UI/test_tools_settings_window.py -- resolver-map coverage, evals/rag resolver-vs-application agreement, a parametrized real backup->restore round trip for all 6 databases, and two failure-mode tests. Verified every new test fails against the pre-fix code before trusting them.

Not fixed (out of scope, flagged for a follow-up task): the "vacuum/backup/check ALL databases" workers and the conversation/character export workers have their own, separate copies of the same hardcoded/profile-unaware literals. Also ChatbookImporter key-casing mismatch, pre-existing and unchanged.

Follow-up (same session, post-review): three more bare self.call_from_thread(...) calls were found in _import_chatbook_worker -- fixed identically. Added test_no_bare_call_from_thread_calls_in_tools_settings_window as a source-scan regression guard.

--- Review follow-up (4 findings addressed post-Done) ---

1. _restore_single_worker's parent-directory guard was too strict: it hard-refused any restore whose target directory did not already exist, blocking a legitimate first restore to a freshly-configured custom database path. Replaced the `if not db_path.parent.exists(): error; return` guard with `db_path.parent.mkdir(parents=True, exist_ok=True)` inside a try/except OSError, matching DB/base_db.py's own behavior when it opens a database. Only a genuine mkdir failure (permissions, invalid path) is now a hard error; a merely-missing directory is not. The `db_path is None` (unresolvable db_name) early return is unchanged and still a hard error. Added test_restore_creates_missing_target_directory_for_a_custom_db_path; verified it fails against the pre-fix guard (reverted, ran, saw the exact old error message, restored, reran green).

2. Added a `_validate_maintenance_path()` helper (Tools_Settings_Window.py) that expanduser()s then routes through Utils/path_validation.py's validate_path_simple() before any backup/restore path reaches shutil.copy2/open/mkdir -- applied to db_path (source) and the computed backup_path (destination) in _backup_single_worker, and to db_path (target) and the user-selected backup_path (source) in _restore_single_worker. On rejection it notifies severity="error" naming the offending path and reason and returns None; callers early-return, never a silent no-op or unhandled exception. validate_path_simple (not the base-directory validate_path) is used deliberately because every real resolved DB path lives under a dotted directory (~/.local/share/...), and validate_path_simple has no hidden-component check to false-positive on that. Added test_restore_refuses_a_dangerous_backup_path_via_path_validation, which creates a REAL file at a semicolon-containing path first (so an unvalidated restore would otherwise succeed) and asserts the rejection names both "dangerous pattern" and the offending path; verified it fails (silently "succeeds" instead) with the validation calls removed, then passes again restored.

3. get_evals_db_path() and get_rag_indexing_db_path() docstrings were missing a Google-style Returns: section (present on their sibling resolvers e.g. _config_write_mode). Added, matching the surrounding style.

4. The new backup/restore round-trip test (and the two new revert-checked tests above) opened sqlite3 connections and called .close() manually, leaking the connection on a mid-test assertion failure. Switched every sqlite3.connect(...) the new tests open to `with closing(sqlite3.connect(...)) as conn:` (contextlib.closing, not a bare `with conn:`, since Connection.__exit__ only commits/rolls back a transaction and does NOT close the connection -- confirmed empirically). Applies to all three connections in the round-trip test plus both new tests.

Modified files: tldw_chatbook/UI/Tools_Settings_Window.py, tldw_chatbook/config.py, Tests/UI/test_tools_settings_window.py. Test run: `.venv/bin/python -m pytest Tests/UI/test_tools_settings_window.py -q` -> 25 passed, 6 pre-existing unrelated chat_api_key failures (identical on dev), 16 skipped (AppTest unavailable in this Textual version).
<!-- SECTION:NOTES:END -->
