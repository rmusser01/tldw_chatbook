---
id: TASK-927
title: Bulk DB maintenance workers carry the same wrong-path bug TASK-899 fixed
status: Done
assignee: []
created_date: '2026-07-27 09:00'
updated_date: '2026-07-27 16:01'
labels:
  - settings
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-899 fixed `Tools_Settings_Window._get_database_path()` so single-database vacuum, backup, restore and integrity-check resolve through `config.py`'s profile-aware resolvers instead of hardcoded literals.

The **"all databases"** variants of those workers were left untouched and do not go through `_get_database_path()`. They carry their own copies of the same hardcoded paths, so they inherit the identical defect: wrong filenames (`tldw_evals_db.db`, `tldw_prompts_db.db`, `tldw_media_db.db` rather than `evals.db`, `tldw_chatbook_prompts.db`, `tldw_chatbook_media_v2.db`) and no profile directory segment.

The consequence is the same one TASK-899 documented. Vacuum, backup and check guard on `exists()` and therefore silently do nothing. Any restore-style path that writes unconditionally would write to a phantom location while the real database is untouched.

This is the more dangerous half in practice: "back up all databases" is exactly what a cautious user clicks before an upgrade.

The conversation/character export workers were also observed to build database paths independently and should be checked in the same pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The bulk workers resolve every database through the same resolvers the single-database workers now use
- [x] #2 No hardcoded database filename or path literal remains in `Tools_Settings_Window.py`
- [x] #3 A bulk backup produces files for the databases that actually exist, proven by a test
- [x] #4 The conversation/character export workers are audited in the same pass and either fixed or explicitly cleared
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read TASK-899's fix and the resulting _DB_PATH_RESOLVERS / _get_database_path in Tools_Settings_Window.py to reuse the same canonical mechanism.
2. Locate all 15 remaining hardcoded ~/.local/share/tldw_cli/<literal>.db occurrences across _vacuum_worker, _backup_worker, _integrity_worker, the three export workers, and _compose_database_config_form / _reset_database_config_form.
3. Add a shared _resolve_legacy_db_targets(db_config) helper resolving the ChaChaNotes/Prompts/Media triad through _DB_PATH_RESOLVERS, returning (resolved, unresolved_display_names); rewrite _vacuum_worker/_backup_worker/_integrity_worker to use it and to report unresolved databases loudly instead of silently proceeding.
4. Add unresolved-path guards (matching the single-database workers' style) to the three export workers.
5. Add a _resolved_db_path_display(db_name) helper and use it in _compose_database_config_form and _reset_database_config_form so the settings form shows the actual resolved path, not a wrong literal, while still round-tripping an explicit custom override unchanged.
6. Write tests proving: bulk workers operate on resolved (not literal) paths; bulk workers/export workers fail loudly on an unresolvable database; a bulk backup produces a file for an existing database at its resolved path; the form shows the resolved default and preserves a custom override. Revert each change individually and confirm the corresponding test fails before restoring.
7. Verify grep for the old literal pattern returns nothing and the existing test suite has no new regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extended TASK-899's fix from the single-database workers to the eight remaining call sites that carried their own copies of the same hardcoded, profile-unaware literals.

Bulk workers (_vacuum_worker, _backup_worker, _integrity_worker): added a shared `_resolve_legacy_db_targets(db_config)` helper that resolves the ChaChaNotes/Prompts/Media triad through the existing `_DB_PATH_RESOLVERS` map (the same one TASK-899 introduced), returning `(resolved, unresolved_display_names)`. All three workers now iterate `resolved` instead of building paths inline, and report `unresolved` loudly (severity="error", naming the databases) instead of silently omitting them while still claiming overall success. `_backup_databases` (the async wrapper) does the resolution check up front and refuses before starting any worker if a path can't be resolved; `_backup_worker` re-resolves as a cheap redundant safety net that raises (falling through to the existing generic failure path) if that ever trips.

Export workers (_export_conversations_worker, _export_notes_worker, _export_characters_worker): replaced the inline ChaChaNotes literal with `self._get_database_path("chachanotes", db_config)` and added an unresolved-path guard matching the single-database workers' style (notify severity="error", return) before the existing exists()-gated flow.

Settings form (_compose_database_config_form, _reset_database_config_form) -- the pair called out as highest-risk, since a save handler at line ~4732 persists these Input values back into config: added `_resolved_db_path_display(db_name)`, which calls `_get_database_path` and returns the actual resolved path as a string. Both functions now use it instead of `db_config.get(key, <wrong literal>)` / a hardcoded literal. Because config.py's resolvers (e.g. `get_chachanotes_db_path`) already return an explicitly-configured custom override unchanged and only fall back to the profile-aware default otherwise, this both fixes the wrong-default display and preserves a real override's round-trip through save/reset -- no second mechanism was invented.

grep -nE '~/\.local/share/tldw_cli/[^"]+\.db' tldw_chatbook/UI/Tools_Settings_Window.py now returns nothing (was 15 matches across 8 functions before this change).

Tests: added 10 tests to Tests/UI/test_tools_settings_window.py -- resolved-vs-literal proof for the bulk vacuum worker (creates a real, padded database only at the resolved path and asserts it shrinks), fail-loud tests for vacuum-all/integrity-all/backup-all/export-conversations with an unresolvable database, a bulk-backup-produces-a-file test (AC #3, also proves the resolved path is targeted), form-shows-resolved-default tests for compose and reset, a custom-override-preservation test, and a source-scan guard that both form functions call `_resolved_db_path_display`. Every new behavior was verified by temporarily reverting the corresponding code, re-running the test, confirming it failed for the right reason, then restoring the fix -- documented per test in the session. One test (`test_export_conversations_fails_loudly_for_unresolvable_chachanotes`) initially passed against reverted code for the wrong reason: removing the guard let `chachanotes_path.exists()` raise `AttributeError` on `None`, which the worker's outer generic `except Exception` handler caught and reported as `severity="error"` anyway -- a real instance of the "masquerading failure" pattern this session was warned about. Fixed by asserting the notification text contains "no resolvable path" (the deliberate guard's message) and does not contain "NoneType" (the crash's message); re-verified the strengthened test now fails correctly against the reverted code.

TLDW_CONFIG_PATH (set per-test by Tests/conftest.py's autouse isolate_test_environment fixture) always wins over the config-file-based override that `mount_settings_window`'s `config_dict` parameter writes and monkeypatches onto DEFAULT_CONFIG_PATH -- a constraint already documented in this file from TASK-899. The custom-override-preservation test therefore shadows `window._DB_PATH_RESOLVERS` at the instance level (the same established workaround `test_restore_creates_missing_target_directory_for_a_custom_db_path` uses) rather than trying to inject a config-file override that would be ignored.

Test run: `.venv/bin/python -m pytest Tests/UI/test_tools_settings_window.py -q` -> 35 passed, 6 pre-existing unrelated chat_api_key failures (verified identical on dev before this session), 16 skipped (AppTest unavailable in this Textual version).

Modified files: tldw_chatbook/UI/Tools_Settings_Window.py, Tests/UI/test_tools_settings_window.py.
<!-- SECTION:NOTES:END -->
