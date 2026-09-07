---
id: TASK-15481
title: Retire dead schedulers and dead DB modules
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - cleanup
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the latency audit — modules that look alive (and cost every future audit/reader time) but are unreachable in production: `Notes/auto_sync_manager.py` (1 s wake-up loop + watchdog, never instantiated; `app.py:9767-9769` stops a field never assigned), `Notes/sync_service.py:436` auto-sync loop (would run sync on the loop; `create_profile` has no production caller), `app.py:9755-9760` stops `self._subscription_scheduler`, also never assigned, `DB/Mindmap_DB.py` (no callers; calls a nonexistent `self.get_connection()` — would AttributeError at `:122/:129`), `DB/search_history_db.py`, `DB/Research_DB.py`, `DB/Writing_DB.py`, `DB/Sync_Client.py` (`ClientSyncEngine` never constructed), and `Widgets/prompt_selector.py` (no non-test importers; its on_mount would issue up to 501 sequential sync queries on the loop if ever wired).

Per the owner's long-term-stability preference: delete (with git-log provenance recorded) or explicitly quarantine each — leaving loaded-gun code that a future contributor wires up IS the instability. Verify each is still dead at implementation time (lessons-backlog-hygiene: verify a reported state still exists before acting on it). Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed module is removed (provenance in notes) or explicitly quarantined with a test asserting non-construction
- [x] #2 app.py no longer stops fields that are never assigned
- [x] #3 Full targeted suite green; no remaining runtime references (grep evidence)
<!-- AC:END -->

## Implementation Plan

1. Re-verify each named item is still dead at HEAD (fdffc031a) via grep for
   production imports, not test imports. Confirmed all named items are dead;
   `Notes/sync_service.py` itself is ALIVE (`NotesSyncService.sync_folder` is
   called from `library_screen.py`) -- only `create_profile` and the
   `start_auto_sync`/`auto_sync_loop`/`stop_auto_sync`/`stop_all_auto_syncs`
   machinery it feeds are dead (no production caller anywhere).
2. Delete standalone-dead modules with zero non-test importers:
   `Notes/auto_sync_manager.py`, `DB/Mindmap_DB.py`, `Widgets/prompt_selector.py`,
   `DB/Writing_DB.py`, `DB/Research_DB.py`, `DB/search_history_db.py`,
   `DB/Sync_Client.py` -- one commit each, deleting their test-only importers
   alongside (or trimming the dead-only slice of a test file that also covers
   live code, e.g. `test_stt_provenance_persistence.py`).
3. Remove the `app.py` teardown blocks that stop `_subscription_scheduler`
   and `_auto_sync_manager`, both never assigned anywhere.
4. Trim `Notes/sync_service.py`: delete `create_profile`, `start_auto_sync`
   (incl. nested `auto_sync_loop`), `stop_auto_sync`, `stop_all_auto_syncs`,
   and the now-unused `_auto_sync_tasks` dict; fix `delete_profile`'s call
   into the removed `stop_auto_sync`. Leave `SyncProfile`/profile load-save/
   `get_profile`/`list_profiles`/`sync_with_profile` in place -- not named
   dead by the task, and not a crash-on-touch landmine like the deleted parts.
5. Update the registries that assert against the deleted DB modules by path:
   `DB/private_sqlite.py`'s `_SQLITE_OWNER_POLICIES` (drop `db.search_history`,
   `db.sync_client_example`), `backlog/docs/sqlite-private-owner-inventory.md`
   (drop the C/P rows for those two owners, renumbering the C table to stay
   contiguous per its own stable-IDs test; P table already tolerates gaps --
   precedent: P24 is already missing), and
   `Tests/DB/test_private_sqlite_inventory.py` (`EXPECTED_PARENT_CREATORS` and
   the two `test_task_*_parent_creators_are_recorded_as_migrated` id lists).
6. Update `CLAUDE.md`'s Data Layer module list to drop the five deleted DBs.
7. Run `pytest --collect-only` over the whole tree (0 errors), then the
   targeted suites for every touched production and test file. Fix forward
   until green.

## Implementation Notes

Every listed item was re-verified dead at HEAD (fdffc031a) before deletion
via production-only import greps (never trusting the audit's file:line cites
without re-checking); all nine were confirmed still dead, so nothing needed
the "leave it, document why" fallback. One item was narrower than its
one-line description implied: `Notes/sync_service.py` itself is alive
(`NotesSyncService.sync_folder` is the Library screen's real notes-sync
path) and was trimmed, not deleted -- only `create_profile` and the
auto-sync-loop machinery it alone could start were removed.

Nine commits, each with `git log --oneline -3` provenance in its body:
1. `Notes/auto_sync_manager.py` deleted + its coalescing test + the two
   `app.py` teardown blocks for `_auto_sync_manager`/`_subscription_scheduler`
   (both fields never assigned anywhere in the codebase).
2. `DB/Mindmap_DB.py` deleted (zero importers; also broken -- called
   `self.get_connection()`, which `BaseDB` never defines).
3. `Widgets/prompt_selector.py` deleted (zero importers anywhere).
4. `DB/Writing_DB.py` + its dedicated test deleted (the live Writing feature
   uses `Writing_Interop/local_writing_service.py`, a separate implementation
   with zero cross-imports).
5. `DB/Research_DB.py` deleted, along with
   `Tests/Research_Interop/test_research_scope_service.py` -- a file whose
   name and directory shadow the real, actively-maintained
   `Tests/Research/test_research_scope_service.py`, but every one of its 8
   tests actually exercised `ResearchDatabase` as a duck-typed stand-in for
   `LocalResearchService`'s dead "external db" delegation branch (production
   always passes a path, never a db object). Coverage of the live,
   path-backed behavior is unaffected (`Tests/Research/*`, untouched).
6. `DB/search_history_db.py` deleted + its dedicated test + its
   parametrized case in `Tests/DB/test_core_sqlite_owner_privacy.py`, plus
   the `db.search_history` entry in `DB/private_sqlite.py`'s owner registry
   and row C16/P15 in `backlog/docs/sqlite-private-owner-inventory.md`
   (renumbering C17-C42 down to C16-C41 to keep the C-table's IDs
   contiguous, per its own test; P15 left as a numbering gap, matching the
   file's pre-existing P24 gap). `Tests/DB/test_private_sqlite_inventory.py`
   updated in step (`EXPECTED_PARENT_CREATORS`, the C-table docstring/range,
   the backup-helper id set, the memory-classification id tuple, and the
   `test_task_four_*` id list).
7. `DB/Sync_Client.py` deleted + its two dedicated tests (`ClientSyncEngine`
   is never constructed in production). `Tests/Media_DB/
   test_stt_provenance_persistence.py` kept its 6 independent MediaDatabase
   provenance tests and lost only the 2 that exercised
   `ClientSyncEngine._apply_remote_changes_batch`. Same registry/inventory
   treatment as step 6, for `db.sync_client_example`/P27.
8. `Notes/sync_service.py` trimmed: removed `create_profile` (no production
   caller) and the auto-sync loop it alone could start
   (`start_auto_sync`/nested `auto_sync_loop`/`stop_auto_sync`/
   `stop_all_auto_syncs`), the now-unused `_auto_sync_tasks` dict and
   `import asyncio`, and fixed `delete_profile`'s now-dangling call into the
   removed `stop_auto_sync`. Left `SyncProfile`, profile load/save,
   `get_profile`/`list_profiles`/`sync_with_profile` in place -- not named
   dead by the task and not a crash-on-touch landmine.
9. `CLAUDE.md`'s Data Layer module list updated to drop the five retired DBs
   (separate doc commit), plus a small doc-freshening commit updating
   `Tests/DB/test_pragma_settings.py`'s "task-15481 will retire them" note
   to past tense.

**Verification.** `pytest --collect-only` over the whole tree: 38785 tests
collected, 0 errors. Targeted suites run to completion in the foreground
(no background/monitor waits used for the final gate, per reviewer
correction): `Tests/DB/` (832 passed, 32 failed -- the standing pre-existing
ChaChaNotes-migration-suite/fixture-shaped failures named in the task
brief), `Tests/Media_DB/` (79 passed), `Tests/Notes/` (1347 passed, 2
skipped), `Tests/Research/`+`Tests/Research_Interop/`+`Tests/Sync_Interop/`
(264 passed), `Tests/ProductionApp/test_service_composition_lifecycle.py`+
`Tests/UI/test_screen_navigation.py` (129 passed, 1 failed + 2 errors),
`Tests/UI/test_library_shell.py` (part of a full combined run: 3203 passed,
34 failed total across every suite above, 3 skipped, 2 errors). A focused
green re-run of every file this task actually edited --
`Tests/DB/test_private_sqlite_inventory.py`,
`Tests/DB/test_core_sqlite_owner_privacy.py`, `Tests/DB/test_pragma_settings.py`,
`Tests/Media_DB/test_stt_provenance_persistence.py`,
`Tests/Notes/test_library_notes_sync_integration.py` -- passed 174/174.

Four failures/errors were not on the task's own standing pre-existing list
(`test_production_app_composes_one_stable_dependency_graph`,
`test_production_app_scheduler_worker_settles_without_contract_error`,
`test_action_library_notes_files_back_returns_to_database`,
`test_library_shell_rail_search_submit_runs_search_canvas_query`), so each
was reproduced in an isolated `git worktree add <path> fdffc031a` (repo's
own `.worktrees/`, never `/tmp`; removed immediately after each check) --
all four fail identically on the unmodified base commit, confirming none
are caused by this task's changes.

Final grep sweep for the nine retired names (module paths, class names,
`app.py` fields, `sync_service.py` dead methods) across `tldw_chatbook/`
and `Tests/` returns zero production hits; the only remaining textual
matches are (a) the intentionally historical/frozen note in
`test_pragma_settings.py`, and (b) an unrelated, pre-existing "Library
search history" feature (`library_screen.py`'s `update_search_history` /
`_library_search_history`, and the `search_history.db` filename used only
as an illustrative example in `sensitive_paths.py`'s docstrings and its
own tests) that merely shares a substring with the deleted
`search_history_db.py` module name.

No lesson filed: nothing here surfaced a new generalizable trap beyond
what `lessons-testing-evidence.md`'s "measure a dead-code graph from both
ends" and "match paths exactly, never by substring" entries already cover
(the `Research_Interop` test-file-name-shadow and the `search_history.db`
substring match were both instances of those existing lessons, not new
ones).

**2026-08-13 review follow-up.** Open task-472 (Onboard prompt_selector UI
analysis-prompt templates to the Internal Prompts registry) explicitly
depends on the ~30 templates that lived in the deleted
`Widgets/prompt_selector.py`. Cross-referenced provenance in both
directions: task-472's Description now points back at commit `0ddd7286c`
and the recovery command
(`git show fdffc031a:tldw_chatbook/Widgets/prompt_selector.py`, verified
to still print the 547-line pre-deletion file), and two stale docs that
still described the deleted modules as live tooling were annotated with
the same retirement/recovery pointers:
`Docs/Parity/2026-04-21-capability-matrix.md:67` (cited
`prompt_selector.py` as existing prompt tooling) and
`tldw_chatbook/DB/DATABASE_PATH_STANDARDIZATION.md:50` (listed
`search_history_db.py` as a completed path-standardization migration).
