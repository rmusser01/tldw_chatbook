# S02 — Notes B (templates, importers, remainder)

**Coverage:** files read in full: 6 | sampled: 12 | mechanical only: 3 (of 21).
Full: `note_import_parsers.py` (982), `note_import_discovery.py` (1325), `note_import_windows_fs.py` (955),
`note_import_executor.py` (1941), `template_store.py` (51), `__init__.py`. Mechanical only: `note_folder_models.py`,
`notes_device_state_schema.py`, `recovery_review.py` (ruff `F,B,E722,ARG,PERF` + pattern greps).

## Findings

### P1 [D1] — 11 of 14 `NotesScopeService` async methods run synchronous SQLite on the event loop; 3 offload
- Where: `Notes/notes_scope_service.py:1322 save_note`, `:1483 get_note_for_sync`, `:1504`, `:1541`, `:1571`,
  `:1610`, `:1639 delete_note`, `:1673 restore_note`, `:1873 search_notes`, `:1929 list_notes`, `:2276
  get_note_detail`. Offloaded siblings: `:1969 list_deleted_notes`, `:2013 list_note_backlinks`, `:2053 count_notes`.
- Evidence: AST scan (`ast.AsyncFunctionDef` × `self.local_notes_service` access × `to_thread|run_finite_local_worker`)
  → `LOCAL-touching, OFFLOADED: 2` / `LOCAL-touching, ON THE LOOP: 10`. `list_notes`/`count_notes` alias the service
  to a local var so the scan missed them; read confirms `list_notes:1958` is on the loop and `count_notes:2100` uses
  `asyncio.to_thread`. `NotesInteropService.list_notes` is `def`, not `async def` (`Notes_Library.py:424`), ending in
  `db.list_notes(...)`.
- **Aggravating detail:** `list_notes:1949` guards its connection pre-open with
  `current_thread() is not main_thread()` — **a negative predicate**. The awaiting caller *is* the main thread, so
  `db` stays `None`, the `finally` close is a no-op, and the branch is dead in production **while the sync query runs
  anyway.** (This is the "negative-predicate thread offload" shape the review prompt names in D1.)
- Why it matters: `search_notes` is an FTS5 `MATCH` and `save_note` is a multi-statement write transaction; both
  freeze the Textual event loop for their whole duration. The three offloaded siblings prove the intended shape
  exists in this same file.
- Recommended correction: route every local branch through the file's own
  `asyncio.to_thread(run_finite_local_worker, ...)` (already at `:487 _run_folder_repository`). **Delete** the
  `current_thread() is not main_thread()` guard rather than inverting it — offload makes it moot.
- Size: M · Confidence: verified
- Pinning test: none (`Tests/Notes/test_notes_scope_service.py` uses async fakes, so it cannot observe loop blocking)
- Already covered: adjacent to TASK-32804.12 (To Do) — **a concrete 11-method instance in one class that .12 does
  not name.**

### P1 [D2] — every note save reads the entire `keywords` table into Python and scans it in O(n·m)
- Where: `Notes/Notes_Library.py:1382-1397` (`_keyword_identity_conflict`), called **unconditionally** at `:1099`
  inside `save_note_with_organization`.
- Evidence: `rows = cursor.execute("SELECT keyword FROM keywords WHERE deleted = 0").fetchall()` — no `LIMIT`, no
  predicate on the requested keywords. Then a nested `any(any(current.casefold() == requested.casefold() …))` where
  `.casefold()` on `current` is **recomputed inside the inner loop** — a note with 10 keywords against 5,000 stored
  keywords is 50,000 `casefold()` calls plus a full table read, **per save**.
- Why it matters: compounds with the finding above — this runs on the event loop. The predicate is one indexed
  query, and it is exactly the `COLLATE NOCASE` form the sibling module already uses
  (`notes_organization_repository.py:1398`). · Size: S · Confidence: verified

### P1 [D1] — `delete_workspace_note` takes a required `version` and silently discards it; its two siblings honour theirs
- Where: `Notes/server_notes_workspace_service.py:727-735`:
  ```python
  async def delete_workspace_note(self, workspace_id, note_id, version) -> dict[str, Any]:
      self._enforce_policy(self._note_action_id("delete", "workspace"))
      client = self._require_client()
      return await client.delete_workspace_note(workspace_id, note_id)   # version dropped
  ```
- Evidence: `ruff --select=ARG` → `ARG002 Unused method argument: version`. `tldw_api/client.py:2595` signature is
  `(self, workspace_id, note_id)` — **no version parameter, so the DELETE goes out unconditional.** Sibling
  `delete_server_note` (`:530`) forwards it: `client.delete_server_note(note_id, expected_version=version)`. Caller
  `notes_scope_service.py:1639 delete_note` passes a real version down all three branches; the local branch honours
  it (`soft_delete_note(user, note_id, version)`), the server branch honours it, **the workspace branch does not.**
- Why it matters: **data loss.** A workspace note edited on another device between the client's read and the user's
  delete is deleted with no 409 — and the signature tells every caller the opposite.
- Size: M (L if the server has no conditional-delete endpoint) · Confidence: verified
- Pinning test: `Tests/Notes/test_notes_scope_service.py:280` and `Tests/Research_Workspace/test_quick_notes.py:1202`
  define fakes with `(workspace_id, note_id, version)` — **they pin the signature, not the forwarding**, so they stay
  green either way.

### P1 [D1] — workspace source and artifact deletion are gated by the *update* policy action, not *delete*
- Where: `server_notes_workspace_service.py:788-791 delete_workspace_source` and `:926-929
  delete_workspace_artifact` both call `self._enforce_policy(self._workspace_action_id("update"))` →
  `notes.workspace.update.server`. `:631 delete_workspace` correctly uses `_workspace_action_id("delete")`.
- Evidence: `runtime_policy/registry.py:376 _resource("notes.workspace", actions=CRUD_ACTIONS)` and `:192
  CRUD_ACTIONS = (LIST, DETAIL, CREATE, UPDATE, DELETE)` — **a distinct `notes.workspace.delete.*` action exists and
  is what `delete_workspace` asks for.**
- Why it matters: a runtime policy that grants workspace update but denies delete **still permits deleting every
  source and every artifact** from a workspace. Two of the three workspace delete paths escape the gate built for
  them. · Size: S · Confidence: verified
- *(Cross-slice: S24 found the same defect class — the gate is applied per-method across 51 copies, and 24 public
  methods never reach it at all. This is the same root cause at a different site.)*

### P1 [D1] — Obsidian `[[wikilinks]]` are rewritten only on the Create path; "Update existing" stores them unresolved
- Where: `Notes/note_import_executor.py:1312-1316` (the single call) vs the update branch at `:1485`, `:1498-1502`.
- Evidence: `grep -rn "rewrite_wikilinks" tldw_chatbook --include=*.py` → **exactly one production call site**, and
  it sits inside `if item.selected_action is ImportAction.CREATE_NEW:`. The update branch passes
  `payload=item.payloads[0]` raw to `replace_note` and compares with `_note_matches(note, item.payloads[0])` raw.
  `note_ids_by_wikilink`/`note_titles_by_wikilink` are built for the whole plan (`:1013-1014`) and handed to
  `_execute_item`, but the else-branch never reads them.
- Why it matters: the same vault file imported as "Create new" gets working `[[target|Title]](note://id)` links and a
  `note_links` row; imported as "Update existing" it gets **literal `[[target]]` text and no backlinks**. The user
  cannot tell which they got.
- Recommended correction: apply `rewrite_wikilinks` once, **before** the action branch, so both paths write and
  compare the same body. (`creatable_wikilink_keys` already excludes UPDATE items from being link *targets* —
  orthogonal and correct.) · Size: S · Confidence: verified (call-site enumeration)
- Pinning test: none. `Tests/Notes/test_note_import_executor.py:1059` covers CREATE_NEW; `:1127` covers "no recorded
  wikilinks". **No test exercises UPDATE_EXISTING + `replace_content=True` + wikilinks.**

### P2 [D4/D1] — `Notes/template_store.py` writes the user's note-template file without fsync and does not use the shared atomic-write helper
- Where: `Notes/template_store.py:20-51 merge_templates`, via `Backup_Recovery/raw_participants.py:827 _file` and
  `:894 _replace`. `grep -n "fsync" raw_participants.py` → **no match**.
- Why it matters: `merge_templates` is a read-modify-write over **all** templates. A crash after `os.replace` but
  before the page cache flushes leaves a truncated file, and the module's own guard
  (`raise ValueError("invalid_note_templates")`) then refuses it — **every saved note template gone, not just the
  new one.**
- **Cluster status:** a second member of the class S01 filed for `file_notes_service.save_file`, and a missed
  adoption site for TASK-32808.5 (Done). **The right fix is one level up from either call site — fsync in
  `raw_participants._replace` — which closes both at once.** *(Lead's note: S25 filed exactly that, independently,
  from the `Backup_Recovery` side.)* · Size: S · Confidence: verified

### P2 [D2] — a full-table keyword scan is executed unconditionally and thrown away in the common path, and the two collision paths disagree on `deleted`
- Where: `Notes/notes_organization_repository.py:1396-1414` (`_reject_resource_name_collision`), called from `:935`
  (every synced keyword) and `:999` (collections).
- Evidence: the indexed `collision` query runs, then `rows = cursor.execute(f"SELECT … WHERE deleted = 0 ORDER BY
  id").fetchall()` runs **unconditionally**, then `rows` is referenced **only** inside the `if collision is None`
  block.
- **Two defects in one:** (a) **cost** — over a sync batch of N keywords this is N full table reads for nothing;
  (b) **drift** — the indexed query has **no `deleted = 0`**, the fallback scan does. A soft-deleted keyword
  therefore blocks adoption through path 1 and is invisible to path 2, so the same name collides or not depending
  purely on whether the exact-case row happens to exist. · Size: S · Confidence: verified

### P2 [D1] — the only two diagnostics in a 2,427-line service lose all their context; 17 of 20 slice files have none at all
- Where: `notes_scope_service.py:1774-1781` and `:1805-1812` — both
  `logger.exception("Failed to enqueue Sync v2 note …", server_profile_id=…, note_id=…, base_version=…, …)`.
- Evidence: loguru puts those kwargs in `record["extra"]`, and the app's sink (`Logging_Config.py:611`) has **no
  `{extra}`** — all six fields discarded at every sink. **Lead re-verified and sized this repo-wide: 71 such calls
  across 9 packages** (see `phase4-verification.md`). Second half: per-file `grep -c "logger\."` → **17 of 20
  non-empty files have zero diagnostics**; only `Notes_Library` (22), `notes_scope_service` (2),
  `notes_device_state_store` (1) log at all.
- Why it matters: when a Sync-v2 enqueue fails the note is saved locally and never syncs, and the line that is
  supposed to say which note, which profile, which version says none of it. S01 filed 18/20 for its files — **the
  cluster is package-wide, not two slices' worth of oversight.**
- Recommended correction: add `{extra}` to the sink **or** inline the fields. The privacy question needs answering
  first: `Notes_Library.py:296-310` deliberately SHA-256s `user_id`/`note_id` before logging ("Keep correlation
  without persisting caller-supplied identifiers"); these two sites log both raw. · Size: S / M · Confidence: verified

### P2 [D3] — `_build_local_notes_graph` (~190 lines) has zero production callers
- Where: `notes_scope_service.py:1123-~1315`. `grep -rn` → the definition plus two hits in
  `Tests/Research_Workspace/test_quick_notes.py`. The only graph entry point, `get_notes_graph:2405`, calls
  `self._require_server_graph_scope(scope)` **first**, so the local scope is rejected before it could reach the
  builder. ~190 lines of closure-heavy graph code (with an N+1 `get_keywords_for_note` at `:1224`) kept alive solely
  by two tests. · Size: S · Confidence: verified

### P3 [D4] — the Windows import adapter imports 13 private names from the POSIX walker and re-rolls 6 more, with 3 behavioural drifts
- Where: `note_import_windows_fs.py:22-43` imports 13 underscore names from `note_import_discovery`, then
  re-defines `_SelectedPath`, `_DiscoveryState`, `_mode_kind` (`:562`, **byte-identical** to `discovery:683`),
  `_admit_file` (`:764`, **byte-identical** to `discovery:979`), `_close_filesystem_descriptors` (`:843` ≈
  `discovery:1123`), `_identity_from_stat`, `_add_failure`, `_carries_obsidian_marker`, `_validate_sibling_namespace`.
- **Drifts:** (1) POSIX `_inspect_selected_path` maps `NotImplementedError` → `secure_discovery_unavailable`; the
  Windows one (`:540`) maps it → `selection_unreadable` — **different user message for the same condition**.
  (2) POSIX branches on `directory_count == 1`, Windows on `directory_count` truthiness (equivalent today only
  because of an earlier reject).
- Why it matters: **the private/shared split is arbitrary** — the shared half is imported across a module boundary
  while the half that actually carries the security semantics (`_admit_file`'s bounds enforcement, `_mode_kind`) is
  duplicated. Any bound change has to be made twice, and **the non-Windows CI never executes the second copy.**
- Size: M · Confidence: verified

### P3 [D3] — the import `template` column is parsed, bounded, hashed into the approval digest, and never applied
- `note_import_parsers.py:762-771`, `:836-880` accept a `template` key/column; `note_import_plan_models.py:377`
  stores `template_name`; `note_import_planner.py:167` and `note_import_execution_models.py:115` fold it into the
  canonical payload digest. **No read anywhere that writes a note.** A user whose export carries `template` sees it
  validated (a >1,024-char value fails the whole record with "fix the template cell") and then silently dropped —
  and it changes the approval hash, so two otherwise-identical plans differing only in an inert field are
  non-interchangeable. · Size: S · Confidence: verified

### P3 [D3] — a test-only method ships on a production repository class
- `note_import_receipts.py:3013 NoteImportReceiptRepository._test_schema_snapshot`, with `ReceiptSchemaSnapshot`
  exported in `__all__`. · Size: S

### P3 [D3] — `_materialize_folder_link`'s `dataset_id` parameter is vestigial
- `notes_organization_repository.py:1217-1220`; the dispatcher at `:912` passes it, `ruff ARG002` confirms it is
  unread, and the two sibling link materializers do not declare it at all. · Size: S

### P3 [D4] — `Notes/` interpolates SQL table/column identifiers at 4 sites without `DB/sql_validation.py`
- `note_folder_repository.py:2592` (`FROM {table}`); `notes_organization_repository.py:1397/1403` (twice).
  **Every call site passes a module-level literal, so there is no live injection** — but
  `grep -n sql_validation tldw_chatbook/Notes/*.py` → **zero imports package-wide**, and the repo built an
  identifier validator for exactly this shape. · Size: S · Confidence: verified

## Candidate triage
**RETIRED — three premises from the dispatch brief that do not hold, each with evidence:**
- **Jinja2 template rendering / sandbox / autoescape / `|safe`:** `grep -rn "jinja2" tldw_chatbook/Notes/` → **zero
  hits.** The notes template system in this slice is `template_store.py` (51 lines): a JSON read/merge store with
  **no rendering at all.** Repo-wide jinja2 lives in `Evals/`, `Chat/prompt_template_manager.py`,
  `Utils/file_extraction.py`, `Web_Server/serve.py` — none in this slice. The only `_render` in these files
  (`note_import_plan_models.py:309`) is a `re.sub` callback. **The real template finding is durability, not
  sandboxing.**
- **XXE / `_KNOWN_UNHARDENED` membership:** no file in this slice appears in the register.
  `grep -rn "ElementTree\|defusedxml\|xml\." tldw_chatbook/Notes/*.py` → **zero. `Notes/` parses no XML/HTML at all.**
- **Archive extraction:** `grep -rn "zipfile\|tarfile\|extractall\|shutil.unpack" tldw_chatbook/Notes/*.py` → **zero.**

**RETIRED with evidence, and worth recording as a positive:** the importer trust boundary. `note_import_discovery.py`
is descriptor-relative throughout: `O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC` for directories (`:1089`),
`O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC` for the leaf (`:1106`), fail-closed via `_SecureDiscoveryUnavailable` when
either flag is missing; every component is `os.stat(..., dir_fd=, follow_symlinks=False)` + `os.fstat`
identity-matched; the leaf is identity-checked **before open, after open, and after read** on
`(dev, ino, mode, size, mtime_ns, ctime_ns)` (`:1148`), then re-walked lexically a second time
(`_verify_lexical_source_binding:516`). Reads are chunked against `bounds.max_file_bytes` with a `+1` overshoot
probe. `display_path` is validated as a safe relative POSIX path at three layers. **This is the best-hardened
filesystem code in the repo;** the only finding against it is the duplication in the P3 above.

**Other retirals:** `except_exception_return` `note_import_executor.py:892` (inside `_translate_exception`'s guard,
returns a privacy-safe typed error). `fetchall_no_limit` 59 rows — **2 confirmed** (filed above), the rest
`_placeholders(len(chunk))`-bounded, `LIMIT 1`, or recursive CTEs over an already-bounded id set.
`fetchall_dynamic_sql` `notes_device_state_store.py:1231` — constant fragments, parameters bound.
`function_body_import` 51 rows — the 14 `server_notes_workspace_service.py` rows are documented task-285 deferrals
and the `recovery*.py` rows are documented ("imports never open runtime stores"); **confirmed as smell:**
`notes_scope_service.py:1940-1942, 2078-2079` import `threading`, `NotesInteropService` and `CharactersRAGDB` inside
the function purely for `type(x) is Y` identity checks — that is the dead-guard machinery from the P1, and it goes
away with the fix. `legacy_markers` ×9 — prose. `lock_and_execute` ×2 — `Notes_Library`'s single `RLock` guards the
`_db_instances` cache only (43 executes run inside `transaction()` cursors, not under it);
`notes_device_state_store`'s guards only the held-connection list (80 executes go through a per-thread
`transaction()`). `raw_1024x1024` — named resource ceilings. `try_import_guard` ×2 — correct platform/optional guards.
**CONFIRMED:** `_mode_kind` ×2, `_admit_file` ×2, `_close_descriptors` ×2 (folded into the P3);
`_enforce_policy`/`_require_client`/`_require_sync_scope_service` (byte-identical members of the S24 cluster).
**UNVERIFIED:** `_normalized_folder_path` (`note_import_executor.py:1763` vs `notes_sync_authority.py:220`, an S01
file). **Not mine:** the `build_conflict_comparison` pair — `file_notes_conflict_compare.py` is not in this file list.
**No `_maybe_await` call site occurs in any S02 file.**

## D4 observations for repo-wide Phase 3
1. **`atomic_write_json` adoption is incomplete in `Notes/`** — and the right fix is one level up: fsync in
   `Backup_Recovery/raw_participants._replace`, the shared write path for **every** `raw._file` writer, closes both
   this and S01's `file_notes_service.save_file`. Worth a repo-wide census of `raw._file` writers.
2. **`_enforce_policy`/`_require_client` is a 43+47-member cluster, and its drift is in the *action id*, not the
   body.** The bodies are byte-identical (confirmed for the 4 `Notes/` members). What drifts is what each call site
   *passes* — `server_notes_workspace_service.py` gates two destructive deletes with `…update.server` while a
   `delete` action is registered. **Consolidating the function would not have caught that.** Suggest Phase 3 census
   **action-id-vs-verb mismatches** across all 47 scope-service pairs, not just duplicate bodies — this slice
   produced one P1 that way and the shape is mechanical. *(S24 built exactly this audit and found 24 more.)*
3. **`DB/sql_validation.py` has zero importers in `Notes/`** while 4 sites interpolate identifiers.
4. **Structured-logging kwargs are silently dropped repo-wide** — one-line sink fix; **lead sized it at 71 calls
   across 9 packages.**
5. **Platform-adapter duplication as a security-boundary pattern.** `note_import_windows_fs.py` ↔
   `note_import_discovery.py` is a 955/1325-line pair where the shared half is imported and the security-bearing
   half is duplicated. **The non-native copy is never executed in CI, so drift is invisible.** If the repo has other
   POSIX/Windows adapter pairs, the same split is worth checking.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| The P1 event-loop blocking is user-observable (UI stalls during a large `search_notes`/`save_note`) | brief forbids running the app | seed ~5,000 keywords in a scratch profile, save a 10-keyword note, watch for a frame stall; or `loop.set_debug(True)` + `loop.slow_callback_duration = 0.1` |
| `Notes_Library.py:1389`'s cost at realistic scale | not measured | time the full scan + nested casefold against 5,000 keywords × 10 requested |
| Whether the server has a conditional-delete endpoint for workspace notes (decides S vs L on the `delete_workspace_note` fix) | server repo not in scope | check the tldw_server OpenAPI for `DELETE /api/v1/workspaces/{id}/notes/{note_id}` and whether it accepts `If-Match`/`expected_version` |
| `notes_device_state_store.py:1062,1168,1324,1381,1830` — unbounded `fetchall` on sync-state tables | sampled the transaction/connection layer only | `rg -n "list_bindings\(\|list_root_summaries\(\|list_incomplete_operations\(" tldw_chatbook --type py` then read each caller for a bound |
| Whether `_normalized_folder_path` has drifted between its two copies | second copy is an S01 file, not re-reviewed per instruction | `diff <(sed -n '1763,1775p' …/note_import_executor.py) <(sed -n '220,232p' …/notes_sync_authority.py)` |
| The 3 mechanical-only files hold no finding | ruff surfaced only `B904`/`ARG001` on `recovery_review.py:120/248` | read them; `pytest Tests/Notes/test_note_folder_models.py Tests/Notes/test_notes_device_state_store.py -q` |
