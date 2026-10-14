# S02 — Notes B validation

## 1. P1 — 11 of 14 `NotesScopeService` async methods run sync SQLite on the event loop
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Notes/notes_scope_service.py:1929-1968` (`list_notes`), unchanged line numbers for the whole cluster (1322 save_note, 1483 get_note_for_sync, 1639 delete_note, 1673 restore_note, 1873 search_notes, 1929 list_notes, 2276 get_note_detail all match the review exactly). Offloaded siblings 1969/2013/2053 also unchanged.
- Proof: `grep -n "to_thread\|run_finite_local_worker" notes_scope_service.py` → only 6 hits, all inside `list_deleted_notes`/`list_note_backlinks`/`count_notes` (lines 491,2005-2006,2046,2100-2101). `list_notes` (:1929-1963) reads: `if (... and current_thread() is not main_thread()): db = service._get_db(local_user)` then unconditionally `return list_notes(local_user, ...)` (sync call) — the guard is a negative predicate since the awaiting caller (Textual event loop) *is* the main thread, so `db` stays `None` and the sync call still runs directly on the loop. `Notes_Library.py:424` confirms `def list_notes` (not `async def`).

## 2. P1 — every note save reads the entire `keywords` table and scans it O(n·m)
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Notes/Notes_Library.py:1382-1397` (`_keyword_identity_conflict`), called unconditionally at `:1099` inside `save_note_with_organization` (def at :797) — unchanged line numbers.
- Proof: `sed -n '1382,1400p'` shows `rows = cursor.execute("SELECT keyword FROM keywords WHERE deleted = 0").fetchall()` — no LIMIT, no predicate — followed by nested `any(any(current.casefold() == requested.casefold() for current in existing) ...)`, `.casefold()` recomputed inside the inner loop. `notes_organization_repository.py:1398` confirms the indexed `COLLATE NOCASE` sibling form exists.

## 3. P1 — `delete_workspace_note` takes a required `version` and silently discards it
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Notes/server_notes_workspace_service.py:727-734` — identical to review's cited lines. Sibling `delete_server_note` at :530-533 forwards `expected_version=version`.
- Proof: `client.delete_workspace_note(workspace_id, note_id)` at line 734 — no `version` arg passed. `tldw_api/client.py:2595-2600` signature is `(self, workspace_id: str, note_id: int)` — genuinely has no version/expected-version parameter, so the DELETE is unconditional as claimed.

## 4. P1 — workspace source and artifact deletion gated by *update* policy action, not *delete*
- Verdict: CONFIRMED
- Site now: `server_notes_workspace_service.py:788-793` (`delete_workspace_source`) and `:926-931` (`delete_workspace_artifact`) both call `self._enforce_policy(self._workspace_action_id("update"))`; `:631-634` (`delete_workspace`) correctly uses `"delete"`.
- Proof: `runtime_policy/registry.py:192` defines `CRUD_ACTIONS = (LIST, DETAIL, CREATE, UPDATE, DELETE)` and `:376` `_resource("notes.workspace", actions=CRUD_ACTIONS)` — a distinct `delete` action is registered and unused by the two source/artifact deletion methods.

## 5. P1 — Obsidian `[[wikilinks]]` rewritten only on the Create path
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Notes/note_import_executor.py:1312-1316` (the single `rewrite_wikilinks` call, inside `if item.selected_action is ImportAction.CREATE_NEW:`) vs the update branch's `self._target.replace_note(..., payload=item.payloads[0])` at ~:1498 — raw payload, no rewrite.
- Proof: `grep -rn "rewrite_wikilinks" tldw_chatbook/` → exactly one production call (note_import_executor.py:1312), plus definition/docs elsewhere. Update branch reads `item.payloads[0]` directly into `replace_note`.

## 6. P2 — `Notes/template_store.py` writes without fsync, not via the shared atomic-write helper
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Notes/template_store.py:17-51` (`merge_templates`, via `raw._file`/`raw._replace`), `Backup_Recovery/raw_participants.py:827` (`_file`), `:894` (`_replace`).
- Proof: `grep -n "fsync" tldw_chatbook/Backup_Recovery/raw_participants.py` → 0 matches. `template_store.py` performs a full read-modify-write (`data = json.load(stream)` then `json.dump({"templates": templates}, stream, ...)` to a `.tmp` file then `raw._replace`), guarded by `raise ValueError("invalid_note_templates")` if the existing store is malformed — matching the "every saved template gone" claim.

## 7. P2 — full-table keyword scan runs unconditionally and the two collision paths disagree on `deleted`
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Notes/notes_organization_repository.py:1383-1414` (`_reject_resource_name_collision`), called from `:935` and `:999` — unchanged line numbers.
- Proof: read of :1394-1403 shows the indexed `collision` query (`WHERE {column} = ? COLLATE NOCASE AND (sync_id IS NULL OR sync_id <> ?)` — no `deleted` filter) executes first, then `rows = cursor.execute(f"SELECT ... WHERE deleted = 0 ORDER BY id").fetchall()` runs unconditionally on the next line, and `rows` is referenced only inside `if collision is None:`.

## 8. P2 — the only two diagnostics in a 2,427-line service lose all context (sink drops `{extra}`)
- Verdict: CONFIRMED
- Site now: `notes_scope_service.py:1775-1782` and `:1806-1813` (both `logger.exception(..., server_profile_id=..., authenticated_principal_id=..., workspace_scope=..., note_id=..., base_version=..., entity_version=...)`).
- Proof: `Logging_Config.py:611` loguru sink format string is `"{time:...} | {level: <8} | {name}:{function}:{line} - {message}"` — contains no `{extra}` token, so every kwarg loguru stores in `record["extra"]` is silently dropped from the rendered line.

## 9. P2 — `_build_local_notes_graph` (~190 lines) has zero production callers
- Verdict: CONFIRMED
- Site now: `notes_scope_service.py:1123` (def) and `get_notes_graph` at `:2405-2406` (`self._require_server_graph_scope(scope)` called first) — unchanged line numbers.
- Proof: `grep -rn "_build_local_notes_graph" tldw_chatbook/ Tests/` → definition plus exactly 2 hits, both in `Tests/Research_Workspace/test_quick_notes.py` (lines 2556, 2708). No production caller.

## 10. P3 — Windows import adapter imports 13 private names and re-rolls 6+ more, with behavioural drift
- Verdict: CONFIRMED
- Site now: `note_import_windows_fs.py:22-43` imports the 13 underscore names from `note_import_discovery` unchanged; re-defines `_SelectedPath` (:234 vs discovery:173), `_DiscoveryState` (:247 vs discovery:161), `_mode_kind` (:562 vs discovery:683 — exact line match to review), `_admit_file` (:764 vs discovery:979 — exact match), `_close_filesystem_descriptors` (:843, discovery has none by that name), `_identity_from_stat` (:881 vs discovery:1009), `_add_failure` (:794 vs discovery:1224), `_carries_obsidian_marker` (:647 vs discovery:1170), `_validate_sibling_namespace` (:658, discovery-only via import).
- Proof: grep for `secure_discovery_unavailable`/`selection_unreadable` confirms the drift — Windows module raises `_selection_error(bounds, "selection_unreadable")` (windows_fs.py:339,541) where POSIX's `_inspect_selected_path` uses `"secure_discovery_unavailable"` for the analogous `NotImplementedError` path (discovery.py:668,1073).

## 11. P3 — the import `template` column is parsed, hashed into the approval digest, and never applied
- Verdict: CONFIRMED
- Site now: `note_import_parsers.py:762-771` (JSON `template` key), `:836-880` (CSV `template` column) → `template_name` field on the payload (`note_import_plan_models.py:377`) → folded into the canonical digest at `note_import_planner.py:167` and `note_import_execution_models.py:115`.
- Proof: `grep -rn "\.template_name" tldw_chatbook/Notes/*.py` → only the 3 definition/validation/digest sites above; no read in `note_import_executor.py` (the note-write path) at all.

## 12. P3 — a test-only method ships on a production repository class
- Verdict: CONFIRMED
- Site now: `note_import_receipts.py:3013` `NoteImportReceiptRepository._test_schema_snapshot`, `ReceiptSchemaSnapshot` (dataclass at :274, exported in `__all__` at :3066) — exact line match to review.
- Proof: direct read, matches claim verbatim.

## 13. P3 — `_materialize_folder_link`'s `dataset_id` parameter is vestigial
- Verdict: CONFIRMED
- Site now: `notes_organization_repository.py:1217-1223` (signature), dispatcher call at `:912`.
- Proof: `ruff check --select=ARG002 notes_organization_repository.py` → `ARG002 Unused method argument: dataset_id` at line 1220, confirming the parameter is passed but never read in the method body.

## 14. P3 — `Notes/` interpolates SQL identifiers without `DB/sql_validation.py`
- Verdict: CONFIRMED
- Site now: `note_folder_repository.py:2603` (`f"SELECT id FROM {table} WHERE deleted = 0 AND id IN ({placeholders})"`); `notes_organization_repository.py:1397,1403` (`f"SELECT ... FROM {table} WHERE {column} = ? ..."` / the fallback scan) both interpolate `table`/`column`.
- Proof: `grep -n "sql_validation" tldw_chatbook/Notes/*.py` → 0 hits package-wide, confirming no `sql_validation.py` usage guards these f-string identifier interpolations (all call sites do pass module-level literals, so this is a hygiene gap not a live injection, matching the review's own framing).

TOTALS: confirmed=14 fixed=0 wrong=0 demoted=0 promoted=0
