# DB-media-base — tldw_chatbook/DB/{base_db.py, sql_validation.py, Client_Media_DB_v2.py}, 11,932 lines

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| tldw_chatbook/DB/base_db.py | 874 | read in full (1–874) |
| tldw_chatbook/DB/sql_validation.py | 792 | read in full (1–792) |
| tldw_chatbook/DB/Client_Media_DB_v2.py | 10,266 | read in full (1–10266, in 700–1000-line chunks; no range skipped) |
| supporting (not in slice, read for the drift table / call-site tracing only) | — | `DB/{Workspace,AgentRuns,Library_Collections,Subscriptions,Evals,RAG_Indexing,Library_Ingest_Jobs}_DB.py` `_get_connection` bodies; `DB/ChaChaNotes_DB.py` 3360–3600, 3640–3660, 3860–3885, 398–422, 17835–17875, 18468–18505; `DB/Prompts_DB.py` 98–122, 3325–3365; `Media/local_media_reading_service.py` 2848–2872, 3380–3412, 3500–3535, 4855–4890; `Media/media_reading_scope_service.py` 125–135, 1795–1822; `Library/meeting_speaker_rename.py` 355–400; `Backup_Recovery/raw_participants.py` 95–160 |

Repro scripts (kept): `<SCRATCH>/home/repro/media_repro{,2,3,4}.py`, run as `cd $WT && source <SCRATCH>/env.sh && $PY <script>`. Every file-backed repro used a DB under the isolated `$HOME/repro/` — `MediaDatabase` opened cleanly there.

## Findings

### P1 [D1] — A bare DML call on `MediaDatabase`'s held connection leaves an implicit transaction open; every later `transaction()` on that thread silently borrows it, nothing commits, and the writes are rolled back at close (one shipped caller does this)
- Where: `tldw_chatbook/DB/Client_Media_DB_v2.py:1325-1372` (`transaction()` borrows when `conn.in_transaction` is already true and then skips `commit()`), `:1072-1083` (the connection deliberately keeps legacy `isolation_level=''` — "task-22224 EXCEPTION … flipping requires this file's own commit/write-site census first … its own task"), `:5412-5413` and `:5530-5531` (`create_document_version` / `update_keywords_for_media` "Assumes called within an existing transaction" but write DML on `self.get_connection()` with no guard). Shipped bare caller: `tldw_chatbook/Media/local_media_reading_service.py:3525` (`db.update_keywords_for_media(media_id, merged_tags)` in `_materialize_reading_import_row`, no `db.transaction()` anywhere in `import_reading_items` → `_execute_reading_import_job` → that helper: `sed -n '2848,3485p' … | rg transaction\(` hits only four unrelated functions). The other three external callers ARE wrapped (`local_media_reading_service.py:4874-4880`, `Library/meeting_speaker_rename.py:369-394`).
- Evidence: `$PY $HOME/repro/media_repro2.py` → `after bare update_keywords_for_media: in_transaction = True` … `after later add_media_with_keywords: in_transaction = True | mid2 = 2` … `after close+reopen: media rows = 1 (expected 2) | keywords linked to mid1 = ['k1'] (expected ['k1','k2']) | read_it_later rows = 0 (expected 1)`. `media_repro.py` part A: the same for a bare `create_document_version` (`DocumentVersions rows after close+reopen (expected 2 if durable): 1`). `media_repro4.py` part D (the write done on another thread, as an executor would): the main thread's next `add_media_with_keywords` → `DatabaseError: Unexpected error processing media: database is locked` (immediately — WAL read→write upgrade against a live writer does not invoke the busy handler; `PRAGMA busy_timeout` on the connection is 10000 ms, verified).
- Why it matters: after one re-import of a reading list whose URL already exists (`merge_tags` defaults to True), every Media write on that thread for the rest of the session is uncommitted — the second manifestation is every OTHER thread's Media write failing with "database is locked". `MediaReadingScopeService.import_reading_items` (`media_reading_scope_service.py:1799-1817`) runs the sync local service inline via `_maybe_await`, i.e. on the event-loop thread, so the leaked transaction would sit on the UI thread's own connection. No UI/tool surface calls `import_reading_items` today (`rg -n -i "import_reading|reading_import" UI Chat Tools MCP Agents Library Tool_Packs app.py` → 0), which is why this is P1 and not P0; it becomes P0 the day one does.
- Recommended correction: root cause is the documented, still-unfiled Media half of task-22224 (`isolation_level=None` + explicit-BEGIN-only manager; the ChaChaNotes precedent is `ChaChaNotes_DB.py:3504-3514`). Grep of `backlog/tasks` for a task naming `Client_Media_DB_v2` + `isolation_level|22224` → none: file it. Interim S fix at the shared function, not per caller: have `create_document_version` and `update_keywords_for_media` open `with self.transaction() as conn:` themselves (a nested call joins the outer transaction exactly as today, a bare caller now gets BEGIN/COMMIT) — or at minimum `if not conn.in_transaction: raise RuntimeError("caller_transaction_required")`, the guard `base_db._SemanticMutationAuthorization._authorize` already uses (`base_db.py:615-616`).
- Size: S (interim guard) · M (isolation flip + write-site census) · ADR: no · Confidence: verified
- Pinning test: none observed. `Tests/Media/test_media_reading_scope_service.py:773,2262` fake the service; `Tests/Media_DB/*` run `:memory:` databases, which cannot observe a close-time rollback.
- Already covered: none (task-22224 covered ChaChaNotes; the Media docstring defers to "its own task", which does not exist)

### P2 [D1] — `MediaDatabase(..., check_integrity_on_startup=True)` always fails: `__init__` calls `self.check_integrity()`, which the class never defines
- Where: `tldw_chatbook/DB/Client_Media_DB_v2.py:1024-1030` (call), whole class (no `def check_integrity`; `rg -n "def check_integrity" Client_Media_DB_v2.py` → exit 1). The standalone `check_database_integrity(db_path)` at `:9103-9143` is the only integrity check that exists. Compare `base_db.py:851-874`, which `MediaDatabase` does not inherit (see the P3 D4a finding).
- Evidence: `$PY $HOME/repro/media_repro4.py` part E → `DatabaseError: Unexpected database initialization error: 'MediaDatabase' object has no attribute 'check_integrity' | cause=AttributeError(...)`.
- Why it matters: the constructor advertises the flag in its signature and docstring (`:973,982`); the first caller to pass it gets a fatal `DatabaseError` from the `except Exception` at `:1047-1056` on every open. No shipped caller passes it today (`rg "check_integrity_on_startup\s*=\s*True" tldw_chatbook` → 0).
- Recommended correction: implement `check_integrity()` as a 6-line instance method around `PRAGMA integrity_check` on `self.get_connection()` (or delete the flag and the dead branch). S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P2 [D4b] — Three cutoff-timestamp formats are compared against one column written in a fourth; rows soft-deleted on the cutoff's calendar date are skipped by the hard-delete cleanup
- Where: writer `tldw_chatbook/DB/Client_Media_DB_v2.py:2210-2218` (`%Y-%m-%dT%H:%M:%S.mmmZ`, every `last_modified`/`trash_date` in the file); readers `:4351-4354` `hard_delete_old_media` and `:4457-4458` `get_deletion_candidates` compare `last_modified < '%Y-%m-%d %H:%M:%S'` (space separator, no ms/Z; `:4457` also uses `datetime.utcnow()`, deprecated in 3.12); `:9255-9257` `empty_trash` compares `trash_date <= '%Y-%m-%dT%H:%M:%SZ'` (no ms). SQLite compares TEXT lexically: `'T'` (0x54) > `' '` (0x20), so on the cutoff date every stored value sorts AFTER the cutoff.
- Evidence: `$PY $HOME/repro/media_repro3.py` → `cutoff (module format): 2026-08-19 02:27:11 | stored: 2026-08-19T00:00:01.000Z | stored is older by 2:27:10` … `get_deletion_candidates(days_old=30) -> 0 row(s) [expected 1]` … `hard_delete_old_media(days_old=30) -> 0 deleted [expected 1]` … `same predicate, cutoff in the STORED format -> 1 row(s)` … `lexical proof: '2026-08-19T00:00:01.000Z' < '2026-08-19 02:27:11' = False`.
- Why it matters: the shipped cleanup (`app.py:19860,19883` → `run_cleanup_method(db.get_deletion_candidates / db.hard_delete_old_media, cleanup_days)`) lags by up to 24 h and its candidate count disagrees with a same-format query; `empty_trash`'s no-ms form is off by ≤1 s in the other direction. Not user-visible today, but it is storage-reaching drift and the `utcnow()` line will start warning.
- Recommended correction: derive every cutoff through the one writer — e.g. a `_utc_timestamp_str(dt)` classmethod used by `_get_current_utc_timestamp_str()` and the three cutoffs — so there is exactly one format. Canonical home: this file's existing helper (or `Utils/` if ChaChaNotes/Prompts share the `%Y-%m-%dT%H:%M:%S.%f`+`Z` form — out of my slice). S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Media_DB/test_media_db_v2.py:1077` and `Tests/RAG/test_ingestion_indexing.py:351` call `hard_delete_old_media(days_old=-1)` — a negative age that never lands on the boundary date, so they do not pin (or contradict) this.
- Already covered: none

### P2 [D4a] — Every held-connection store re-rolls the PRAGMA/WAL/busy_timeout/isolation setup that `BaseDB._get_connection` could own; the undocumented rows drift
- Where: `tldw_chatbook/DB/base_db.py:818-825` sets only `row_factory`. Overrides (all read):

| store (`_get_connection` / opener) | foreign_keys | journal_mode WAL | synchronous | busy_timeout | isolation_level | documented? |
|---|---|---|---|---|---|---|
| `BaseDB` 818-825 | — | — | — | sqlite3 default 5 s | legacy | — |
| `Workspace_DB.py` 353-388 | ON | file-only | NORMAL | default 5 s | None | task-3012/15480 comment |
| `AgentRuns_DB.py` 297-337 | ON | file-only | NORMAL | `PRAGMA busy_timeout=5000` set BEFORE WAL (only store that orders it) | None | comment |
| `Library_Collections_DB.py` 498-543 | ON | file-only via `_enable_wal` retry loop (only store with one) | NORMAL | default 5 s | None | task-15466 comment |
| `Subscriptions_DB.py` 614-660 | ON | file-only, non-RO | NORMAL | `BUSY_TIMEOUT_MS` | legacy | **task-22224 EXCEPTION** (docstring) |
| `Evals_DB.py` 196-236 | ON | unconditional (also `:memory:`) | NORMAL | default 5 s | legacy | **task-22224 EXCEPTION** (docstring) |
| `RAG_Indexing_DB.py` 104-144 | **absent** (declares no FKs — verified `rg -i "FOREIGN KEY\|REFERENCES"` → 0) | file-only | NORMAL | default 5 s | None | comment |
| `Library_Ingest_Jobs_DB.py` 84-110 (the template) | **absent** (no FKs) | unconditional | NORMAL | default 5 s | None | module docstring |
| `Client_Media_DB_v2.py` 1127-1152 | ON | file-only | NORMAL | connect `timeout=10` (→ `PRAGMA busy_timeout` 10000 verified) | legacy | **task-22224 EXCEPTION** (docstring) |
| `ChaChaNotes_DB.py` 3480-3514 | ON | file-only | NORMAL | `timeout=15` | None | task-22224 comment |
| `Prompts_DB.py` 462-482 | ON | file-only | NORMAL | `timeout=10` | legacy | **task-22224 EXCEPTION** (docstring) |

- Evidence: read only (table above is from the cited ranges); `media_repro4.py` PRAGMA read-back for the Media row.
- Why it matters: the documented `isolation_level` rows are P3 (and the Media one is where the P1 above lives). The undocumented drift is `busy_timeout` (five stores on the implicit 5 s default, three on 10–15 s, two on an explicit PRAGMA) and the WAL-conversion race (handled by ordering in AgentRuns, by a retry loop in Library_Collections, by nothing elsewhere) — the same cross-process first-open contention hits every file store identically. `foreign_keys` absence in RAG_Indexing/Library_Ingest_Jobs is inert today (no FKs) and stays P3.
- Recommended correction: one `configure_held_connection(conn, *, is_memory: bool, busy_timeout_ms: int)` in `DB/base_db.py` (row_factory, `busy_timeout` first, `foreign_keys`, guarded WAL, `synchronous=NORMAL`, `isolation_level=None`) that `BaseDB._get_connection` calls and the four legacy-isolation stores call with an explicit `legacy_isolation=True` until their task-22224 census lands. The template docstring in `Library_Ingest_Jobs_DB.py:1-20` then points at code instead of prose. M.
- Size: M · ADR: no (adr_list grep for connection/pragma/wal/isolation → only 114/125/157, unrelated) · Confidence: verified (table) / inferred (that the WAL race reaches the un-guarded stores in practice — literal check: run two processes opening the same fresh `.db` under `Evals_DB` concurrently)
- Pinning test: none for the shape; `Tests/DB/test_chachanotes_connection_quiescence.py` pins ChaChaNotes' registry behaviour only
- Already covered: task-15466 / task-15480 / task-21101 ported the idiom store-by-store; none owns the shared helper

### P3 [D4a] — `MediaDatabase` does not subclass `BaseDB` and re-implements its path handling, `vacuum`, and `close`
- Where: `tldw_chatbook/DB/Client_Media_DB_v2.py:263` (`class MediaDatabase:`), `:988-1001` (duplicate of `base_db.py:771-783` path/`:memory:` normalization), `:8576-8591` (`vacuum`, cf. `base_db.py:834-849`), `:8572-8574` (`close`), the missing `check_integrity` (P2 above).
- Evidence: read only.
- Why it matters: the "canonical home every other DB module is measured against" is bypassed by the second-largest store, which is how the `check_integrity` call could reference a method that only exists on the base class it does not inherit.
- Recommended correction: inherit `BaseDB` (its `__init__` already does the same normalization; pass `initialize_schema=False` and keep the existing `_initialize_schema`). S/M depending on the `Backup_Recovery` `_core_*` decorators' expectations.
- Size: M · ADR: no · Confidence: verified (structure) · Pinning test: none · Already covered: none

### P3 [D3] — loguru and stdlib `logging` used side by side in Client_Media_DB_v2 (124 stdlib calls, 221 loguru), with one method shadowing the module logger
- Where: `tldw_chatbook/DB/Client_Media_DB_v2.py:33,48` (both imported); stdlib sites e.g. `:1008,1039,1049,1436,2139,2201,4620,5345,7081`; `:5666` (`logger = logging.getLogger(__name__)` inside `update_media_metadata` shadows the loguru `logger` for that method); `:1235-1246` comment explicitly calls the stdlib module "unconfigured -> effectively silent".
- Evidence: `rg -c '\blogging\.(debug|info|warning|error|critical)\('` → 124; `rg -c '\blogger\.'` → 221. `Logging_Config.py:410,712,747` attaches handlers to the stdlib root logger, so the stdlib lines are NOT silent in the app — the comment at `:1235` is stale rather than wrong in effect.
- Why it matters: two formatting/lazy-evaluation regimes in one file (every stdlib call builds its f-string eagerly, including `exc_info=True` migration errors), and the `:5666` shadow means that method's errors skip loguru's sinks/sanitizer.
- Recommended correction: mechanical swap to `logger` (loguru) file-wide; delete `:5666`. S.
- Size: S · ADR: no (`Logging_Config.py` is the one legitimate bridge; this file is not it) · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D3] — `base_db.py` hard-codes its own subclasses and needs 7 function-body imports to dodge the resulting cycle
- Where: `tldw_chatbook/DB/base_db.py:48-58` (`operation_owned_connection` imports `AgentRunsDB`, `LibraryCollectionsDB`, `WorkspaceDB`, `Backup_Recovery.participants._core_cached_connection` and switches on `type(database) not in {...}`), `:89-93` (`run_owned_db_call` same with `CharactersRAGDB`).
- Evidence: read only.
- Why it matters: adding a store to the "owned connection" contract means editing the base module's allowlist; the base depends on its children.
- Recommended correction: a class attribute/flag on `BaseDB` (`_owned_connection_contract = True`) that the four stores set, replacing the `type(...) in {...}` sets and the lazy imports. S.
- Size: S · ADR: no · Confidence: verified · Pinning test: `Tests/Backup_Recovery/test_console_resume_character_lifetime.py`, `Tests/Canvas/test_repository.py` reference the registry/owned path (behaviour, not the allowlist shape) · Already covered: none

### P3 [D4b] — `_library_fts_query` / `_library_keywords_for_*` / `_escape_library_like` are four-way shape clones across the three FTS stores
- Where: `Client_Media_DB_v2.py:8606-8646`, `Prompts_DB.py:3325-3360`, `ChaChaNotes_DB.py:17835-17870` and `:18468-18505` — identical bodies modulo table/column names and the per-class token-limit constant; `ConflictError.__str__` is verbatim ×3 (`Client_Media_DB_v2.py:114-121`, `Prompts_DB.py:107-114`, `ChaChaNotes_DB.py:409-416` with `entity_id`).
- Evidence: read (diffed by eye; the `re.findall(r"\w+")` + `quote_fts5_token` + `" ".join` body is byte-identical apart from the limit name).
- Why it matters: no behavioural drift today; the next FTS5 quoting fix has four places to land.
- Recommended correction: `Utils/fts5_match_forms.py` already exports `quote_fts5_token` — add `and_of_quoted_tokens(raw, limit)` there; the keyword grouper is a 6-line generic `(conn, link_table, id_col, ids)` helper for `DB/base_db.py`; `ConflictError` to a shared `DB/errors.py`. S.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D3] — Dead-in-prod helpers that materialise every media row WITH `content`, and three `not implemented` stubs
- Where: `Client_Media_DB_v2.py:10068-10098` `get_all_content_from_database`, `:9326-9353` `get_unprocessed_media`, `:8434-8461` `get_all_active_media_for_embedding(limit=None)`, `:3366-3533` `fetch_media_for_keywords`, `:9659-9698` `get_media_prompts`, `:7436-7503` `get_all_active_media_for_selection_dropdown`; stubs `:9088-9100` (`create_incremental_backup`, `create_automated_backup`, `rotate_backups` — log a warning and `pass`), `:9319-9322` deprecated `check_media_and_whisper_model`.
- Evidence: whole-repo `rg -n "\b<name>\b" --glob '!DB/Client_Media_DB_v2.py'` → only `Tests/DB/test_pagination.py` / `Tests/Library/test_library_tool_security_bounds.py:407` (a name allowlist) for the first three; zero hits at all for `fetch_media_for_keywords`, `get_media_prompts`, `get_all_active_media_for_selection_dropdown`, the three stubs, and `check_media_and_whisper_model`.
- Why it matters: three of these are unbounded full-`content` scans waiting for a caller; the stubs return success-shaped nothing.
- Recommended correction: delete (tests that only exercise the dead helper go with them). S.
- Size: S · ADR: no · Confidence: verified (grep + collect-only not needed: no re-export exists, `rg "get_all_content_from_database"` over the tree is exhaustive) · Pinning test: `Tests/DB/test_pagination.py:103-115,155-156` state the current unbounded behaviour as a requirement ("doesn't support pagination parameters") — so unbounded is a decision for those two; deletion, not pagination, is the ask · Already covered: none

### P3 [D3] — Function-body imports of modules already imported at module scope, one on the per-call connection hot path
- Where: `Client_Media_DB_v2.py:1099` (`from tldw_chatbook.Backup_Recovery.participants import _core_access` on every `_get_thread_connection` call while its five siblings are imported at `:60-66`), `:4348` (`from datetime import timedelta`), `:4455` (`from datetime import datetime, timedelta`) — both already at `:40`. (`:1899` is documented lazy and a test seam — fine.)
- Evidence: read only.
- Recommended correction: hoist. S.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D1] — `BaseDB.vacuum` / `check_integrity` leak the connection on exception and `check_integrity` returns a non-bool
- Where: `base_db.py:842-849` (`conn.close()` not in `finally`), `:858-874` (same; `return result and result[0] == "ok"` yields `None`/`Row` on the falsy path).
- Evidence: read only.
- Recommended correction: `with contextlib.closing(self._get_connection()) as conn:` and `bool(...)`. S.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

## Candidate dispositions
| candidate (file:line pattern) | confirmed / retired (why) / unverified (check) |
|---|---|
| dup_shape `_library_*_fts_query` ×4 | confirmed — P3 D4b above |
| dup_shape `_library_keywords_for_*` ×4 | confirmed — P3 D4b above |
| dup_shape / dup_verbatim `ConflictError.__str__` ×3 | confirmed — folded into the P3 D4b finding |
| fetchall_dynamic_sql 3142 `search_media_db` | retired — `LIMIT ? OFFSET ?` (3131); paginated |
| fetchall_dynamic_sql 3244 `_library_browse_keyword_only_matches` | retired — bounded by the page's `media_ids` |
| fetchall_dynamic_sql 3465 `fetch_media_for_keywords` | confirmed dead (0 callers) — P3 D3 |
| fetchall_dynamic_sql 3565 `get_sync_log_entries` | retired — optional `LIMIT`; callers are the sync layer (one-shot) |
| fetchall_dynamic_sql 3853 `list_read_it_later_media_ids` | retired — optional `LIMIT`; the one caller (`local_media_reading_service.py:427`) uses it as an id filter for a paginated search; bounded by the user's saves, not media |
| fetchall_dynamic_sql 6266 `get_keyword_usage_stats` | retired — keywords table (small); called via `asyncio.to_thread` (`Chat/scope_picker_listers.py:503`), not on the loop |
| fetchall_dynamic_sql 6824 `get_all_document_versions` | retired — per-media, optional LIMIT; UI caller `local_media_reading_service.py:4832` |
| fetchall_dynamic_sql 7110 `get_paginated_media_list` | retired — LIMIT/OFFSET; UI caller `UI/Wizards/ChatbookCreationWizard.py:373` asks page 1 ×100 |
| fetchall_dynamic_sql 7479 `get_all_active_media_for_selection_dropdown` | confirmed dead (0 callers), LIMIT present — P3 D3 |
| fetchall_dynamic_sql 7571 `get_paginated_files` | retired — LIMIT/OFFSET |
| fetchall_dynamic_sql 7640 `get_all_active_media_ids` | retired — ids only; documented truncation-proof export source (`Library/library_export_scope.py:156,353`) |
| fetchall_dynamic_sql 7726 `get_distinct_media_types` | retired — `DISTINCT type` (tiny); v9 index (`:471-473`) covers it |
| fetchall_dynamic_sql 8455 `get_all_active_media_for_embedding` | confirmed dead (0 prod callers), unbounded when `limit=None` — P3 D3 |
| fetchall_dynamic_sql 8480 `get_media_by_ids_for_embedding` | retired — bounded by the id list (RAG search_service) |
| fetchall_dynamic_sql 8554 `fetch_keywords_for_media_batch` (method) | retired — bounded by the page's ids |
| fetchall_dynamic_sql 8644 `_library_keywords_for_media` | retired — bounded by the page's ids |
| fetchall_dynamic_sql 9348 `get_unprocessed_media` | confirmed dead (0 prod callers), full-`content` scan — P3 D3 |
| fetchall_dynamic_sql 9547 `get_media_transcripts` | retired — per-media (MCP resources/prompts) |
| fetchall_dynamic_sql 9692 `get_media_prompts` | confirmed dead (0 callers) — P3 D3 |
| fetchall_dynamic_sql 10205 `fetch_keywords_for_media` | retired — per-media; UI caller `Widgets/media_details_widget.py:347,525` |
| fetchall_dynamic_sql 10253 `fetch_keywords_for_media_batch` (standalone) | retired — bounded by the id list |
| fetchall_dynamic_sql base_db 528 `fetchall` | retired — it is the cursor wrapper itself |
| fetchall_no_limit 1463 PRAGMA table_info | retired — schema metadata |
| fetchall_no_limit 1929 ChunkingTemplates migration | retired — one-shot migration over a small table |
| fetchall_no_limit 2064 sqlite_master | retired — schema metadata |
| fetchall_no_limit 4007 / 4051 `soft_delete_media` | retired — child rows of ONE media id |
| fetchall_no_limit 4286 `undelete_media` | retired — same |
| fetchall_no_limit 4373 `hard_delete_old_media` | confirmed (format drift, not size) — P2 D4b |
| fetchall_no_limit 4468 `get_deletion_candidates` | confirmed (format drift) — P2 D4b |
| fetchall_no_limit 5549 `update_keywords_for_media` | retired — links of one media id; the function itself is the P1 |
| fetchall_no_limit 5953 `soft_delete_keyword` | retired — links of one keyword |
| fetchall_no_limit 6175 `merge_keywords` | retired — links of the given keywords |
| fetchall_no_limit 7045 `fetch_all_keywords` | retired — Keywords only; UI caller (`list_media_keywords`, `local_media_reading_service.py:583`) then filters in Python — acceptable at keyword scale |
| fetchall_no_limit 8704 / 8774 / 8788 / 8903 COUNT rows | retired — scalar counts |
| fetchall_no_limit 9267 `empty_trash` | confirmed (no-ms cutoff format) — folded into P2 D4b; size bounded by trash |
| fetchall_no_limit 10093 `get_all_content_from_database` | confirmed dead, unbounded with `content` — P3 D3 |
| function_body_import_per_file base_db (7) | confirmed — P3 D3 (cycle-dodging subclass imports) |
| function_body_import_per_file Client_Media_DB_v2 (4) | confirmed — P3 D3 (1099, 4348, 4455; 1899 documented) |
| id_keyed_dict base_db:157 | retired — see Verified-fine (strong ref; identity-guarded unregister; every close path unregisters) |
| legacy_markers_per_file Client_Media_DB_v2 (7) | confirmed as P3 dead stubs (9088-9100, 9319-9322) plus the three documented legacy-isolation notes; exact 7-marker census not re-derived |
| lock_and_execute Client_Media_DB_v2 (locks=2, executes=124) | retired — both locks (`:150-151`) guard the post-ingest/post-delete callback lists; dispatch snapshots under the lock and invokes outside (`:215-223`, `:251-259`); no SQL under either |
| lock_and_execute base_db (locks=3, executes=5) | retired — `_QUIESCENCE_REGISTRIES_LOCK`, the registry `Condition(RLock)` and the connection's token `RLock` guard bookkeeping only; the 5 executes are cursor wrappers and the trace-GC SELECT (`:682-690`), which runs on the caller's cursor with no lock held |
| loguru_and_logging Client_Media_DB_v2 | confirmed — P3 D3 |
| seed_name `_get_connection` base_db:818 | confirmed — anchor of the P2 D4a drift table |
| seed_name `_initialize_schema` ×2 | retired — abstract in base, one real override (Media 2132-2207 migration driver); no duplication |
| strftime 2218 `%Y-%m-%dT%H:%M:%S.%f` | retired — Python `%f` sliced `[:-3]` to ms; correct |
| strftime 4354 / 4458 / 9257 | confirmed — P2 D4b |

## Verified-fine
- **`base_db.py:157` id()-keyed registry.** The dict holds a strong reference, so a registered connection can never be collected and its `id()` can never be reused while the entry exists; `unregister` (159-170) and `is_registered` (172-183) both check `current is connection` before acting, so a foreign object with a recycled id is a no-op. Removers: `unregister` from ChaChaNotes' three close paths (`ChaChaNotes_DB.py:3474` liveness-reopen, `:3558` init failure, `:3877` `close_connection`) and from `close_registered` (250-274), which unregisters an already-closed handle via the `sqlite3.ProgrammingError` branch. `media_repro.py` part C: `after close+del+gc, connection_count = 1 … new connection id == old id ? False … after close_registered, connection_count = 0`. A connection closed WITHOUT `unregister` would leak until the next maintenance window — no such path exists in the readers I traced.
- **`BaseDB` has no `transaction()`/`connection()`** (the excerpt's framing); those live per store. `MediaDatabase.transaction()` is thread-local (`threading.local`, `check_same_thread=False`, WAL, `busy_timeout` 10 s verified) and nesting joins the outer — correct except for the borrow hazard in the P1, which is the documented task-22224 exception.
- **`str(params)` logging** — already lazy: `logger.opt(lazy=True).debug(... preview_params(params))` at `:1243-1246` and `:6818-6821`; `execute_many` logs only `len(params_list)`.
- **`except Exception` on write paths** — all 61 sites re-raise as `DatabaseError` or the specific error, except the documented best-effort ones: callback dispatch (`:220,256`), `backup_database` → `False` (`:7688`), `empty_trash` → `(0, -1)` (`:9312`, docstring says so), and the FTS-table creation in `_apply_schema_v1` (`:1531-1534`, warning-only; an FTS5-less SQLite is not a shipped configuration).
- **`_MEDIA_IDS_FILTER_JSON_EACH_THRESHOLD`** (`:159,2727-2742`), the v8/v9 index measurements (`:355-481`), and the CROSS JOIN count-plan pin (`:3105-3126`) are deliberate and documented; not re-litigated.
- **`_execute_transactional_script`** builds statements with `statement += character` over the migration script (`:1397-1403`) — runs once per migration on a few-KB string; not a hot loop.
- **External `media_db.get_connection().execute(` sites** (`Library/library_rechunk_service.py:140`, `Library/local_media_chunk_tool_service.py:327,347,354`) and all external `execute_query` sites (`RAG_Search/ingestion_indexing.py:1504,1526`, `UI/Screens/library_screen.py:30939`, `Library/library_export_scope.py:304`) are SELECTs — they cannot trip the P1.
- **`sql_validation.py`** — `validate_column_name` fails closed (687-698), `validate_table_name` requires the allowlist, `_SAFE_ORDER_BY_PROFILES` re-validates its own terms; `Tests/DB/test_sql_validation.py` → 26 passed under the isolated env (the one failure is an env artifact, see UNVERIFIED). The media/prompts `VALID_TABLES` sets (186-213) have no live-schema pin — that is task-19867; not re-recommended.

## Retired
- **"PRAGMA foreign_keys missing in RAG_Indexing/Library_Ingest_Jobs is a storage bug"** — retired: neither schema declares a `FOREIGN KEY`/`REFERENCES` (`rg -i` → 0), so the pragma is inert there; kept as a P3 row in the drift table only.
- **"Cutoff drift is benign because a 30d23h-old row was selected"** (my first probe, `media_repro2.py` part B) — the row crossed a calendar-date boundary so the date prefix decided; retired as evidence and re-run with a same-date row (`media_repro3.py`), which reproduced the skip.
- **"`timeout=10` is dropped by `connect_private_sqlite`"** (suspected from the 0.0 s lock failure in part D) — retired: `PRAGMA busy_timeout` reads back 10000 ms; the immediate `database is locked` is WAL's read→write upgrade semantics against a live writer, not a missing busy handler.
- **"stdlib `logging` in Client_Media_DB_v2 is silent in the app"** (the file's own comment at `:1235`) — retired: `Logging_Config.py:410,712,747` attach handlers to the stdlib root; the finding stays P3 consistency, not swallowed diagnostics.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| `Tests/DB/test_sql_validation.py::TestValidateTableName::test_chunking_templates_columns_accepted_and_live` fails ONLY because of the isolated env (`RecoveryRequired('raw_source_selection_changed')` raised from `Backup_Recovery/raw_participants.py:_participant_state` when `MediaDatabase(tmp_path/'cols.db')` opens under pytest's conftest + scratch `TLDW_CONFIG_PATH`), not because of a `ChunkingTemplates` column drift | the brief forbids running against the real profile; my own file-backed `MediaDatabase` under `$HOME/repro/` opened fine, which points at the conftest/config binding rather than the schema, but that is inference | from the user's normal dev shell (main checkout): `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook && .venv/bin/python -m pytest Tests/DB/test_sql_validation.py -q` — expect 27 passed |
| The WAL first-conversion race that `AgentRuns_DB` orders around and `Library_Collections_DB` retries around actually bites the stores that do neither (`Evals_DB`, `Workspace_DB`, `RAG_Indexing_DB`, `Library_Ingest_Jobs_DB`, `Subscriptions_DB`) | needs two processes opening the same fresh file at once; not attempted | `cd $WT && source <SCRATCH>/env.sh && for i in 1 2; do $PY -c "from tldw_chatbook.DB.Evals_DB import EvalsDB; EvalsDB('$HOME/repro/race.db')" & done; wait` — repeat ~20×, look for `OperationalError: database is locked` |
| `MediaReadingScopeService.import_reading_items` running the sync import inline on the event loop (`media_reading_scope_service.py:1799-1817` via `_maybe_await`) is the general local-mode pattern rather than an oversight (a sibling `_is_memory_backed` helper at `:131` suggests other calls go through `to_thread`) | outside my slice; only the two ranges above were read | `rg -n "to_thread|_is_memory_backed" tldw_chatbook/Media/media_reading_scope_service.py` and compare against the `import_reading_items` body |
