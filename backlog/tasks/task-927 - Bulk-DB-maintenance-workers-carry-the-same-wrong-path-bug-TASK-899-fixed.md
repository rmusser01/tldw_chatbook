---
id: TASK-927
title: Bulk DB maintenance workers carry the same wrong-path bug TASK-899 fixed
status: Done
assignee: []
created_date: '2026-07-27 09:00'
updated_date: '2026-07-27 16:10'
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

Settings form (_compose_database_config_form, _reset_database_config_form): added `_resolved_db_path_display(db_name, *, ignore_override=False)`, which calls `_get_database_path`/the resolver and returns the actual resolved path as a string. Both functions use it instead of `db_config.get(key, <wrong literal>)` / a hardcoded literal.

Follow-up (post-review): the coordinator caught that Reset, as first implemented, called the same override-aware _resolved_db_path_display as Compose -- so for a user who had genuinely customized a database path (the one case someone actually reaches for "Reset"), the button did nothing. Fixed by adding an `ignore_override: bool = False` keyword-only parameter to `get_chachanotes_db_path`, `get_prompts_db_path` and `get_media_db_path` in config.py (default False, fully backward compatible with all existing zero-arg callers): when True it skips `get_cli_setting` and always returns the profile-aware default already computed in each resolver's own `else` branch -- reusing the single source of truth for the per-database filename instead of duplicating it in the UI, per the coordinator's explicit direction. `_resolved_db_path_display` gained a matching `ignore_override` parameter, forwarded to the resolver only when True (so Compose's existing zero-arg call path is byte-for-byte unchanged). `_reset_database_config_form` now passes `ignore_override=True` on all three fields; `_compose_database_config_form` was left untouched (still shows the currently-effective, override-aware value).

grep -nE '~/\.local/share/tldw_cli/[^"]+\.db' tldw_chatbook/UI/Tools_Settings_Window.py returns nothing (was 15 matches across 8 functions before this task).

Tests: added 11 tests total to Tests/UI/test_tools_settings_window.py across both commits -- resolved-vs-literal proof for the bulk vacuum worker, fail-loud tests for vacuum-all/integrity-all/backup-all/export-conversations with an unresolvable database, a bulk-backup-produces-a-file test, form-shows-resolved-default tests for compose and reset, a custom-override-preservation test, a source-scan guard that both form functions call `_resolved_db_path_display`, and (follow-up) a test pinning the Compose-vs-Reset distinction that simulates a real custom override by monkeypatching `config.get_cli_setting` itself (special-cased to the one key under test) rather than the config file, since the per-test TLDW_CONFIG_PATH env var always wins over a config-file-based override in this test app. Every new behavior was verified by temporarily reverting the corresponding code, re-running the test, confirming it failed for the right reason, then restoring the fix.

Two "masquerading failure" traps were caught and fixed during revert-checking (both documented as the session was explicitly warned to watch for this pattern):
1. `test_export_conversations_fails_loudly_for_unresolvable_chachanotes` initially passed against reverted code for the wrong reason: removing the guard let `chachanotes_path.exists()` raise `AttributeError` on `None`, caught by the worker's outer generic `except Exception` and reported as `severity="error"` anyway. Fixed by asserting the notification text contains the deliberate guard's message ("no resolvable path") and not "NoneType".
2. The Compose-vs-Reset source-scan guard test initially passed against a reverted Reset because a leftover explanatory *comment* still contained the string "ignore_override=True" even after the actual code no longer used it. Fixed by stripping `#`-comment lines before the substring check.

Test run: `.venv/bin/python -m pytest Tests/UI/test_tools_settings_window.py -q` -> 36 passed, 6 pre-existing unrelated chat_api_key failures (verified identical on dev before this session), 16 skipped (AppTest unavailable in this Textual version). Also ran Tests/test_smoke.py and Tests/Tools/test_note_tool_user_id.py (both call the three modified config.py resolvers) to confirm the added keyword-only parameter didn't break any existing zero-arg caller: 24 passed.

Modified files: tldw_chatbook/UI/Tools_Settings_Window.py, tldw_chatbook/config.py, Tests/UI/test_tools_settings_window.py.
<!-- SECTION:NOTES:END -->
