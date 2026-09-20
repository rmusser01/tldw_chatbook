# DB-chacha — tldw_chatbook/DB/ChaChaNotes_DB.py, 24,180 lines (+ base_db.py 874, migrations/README.md 217 for context)

Worktree `/Users/macbook-dev/Documents/GitHub/tldw-review` @ d8fb4053f9. All commands below were run with `cd <worktree> && source <SCRATCH>/env.sh && $PY ...`; repro scripts are under `<SCRATCH>/repro_*.py`. Nothing in the worktree was modified; no app run; no full-suite run. `ruff check --select E9,F63,F7,F82` on both files: `All checks passed!`.

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| tldw_chatbook/DB/ChaChaNotes_DB.py | 24,180 | **read in full** (1–720 direct; 720–2000 and 2000–3306 = schema/migration SQL literals, read line-by-line with trigger-payload boilerplate collapsed via grep after the first full copy of each trigger family; 3306–24180 read in full in 2000-line chunks) |
| tldw_chatbook/DB/base_db.py | 874 | read in full (only to judge bypass + quiescence tax; another agent owns findings) |
| tldw_chatbook/DB/migrations/README.md | 217 | read in full |
| tldw_chatbook/DB/sqlite_datetime_fix.py | 80 | read in full (needed for the timestamp finding) |
| tldw_chatbook/Utils/fts5_match_forms.py | — | sampled: `fts5_query_tokens`, `quote_fts5_token`, `build_and_match_expression/query`, `build_phrase_match_query` bodies |
| `.sql` migration files | — | NOT read (runner methods read; README says preflight/allowlist checks own them) |

## Findings

### P1 [D1] — A reviewed flashcard is not reported "due" until the UTC day after its due time; study stats drop the boundary day
- Where: `tldw_chatbook/DB/ChaChaNotes_DB.py:21570` (writes `next_review = datetime.now(utc)+timedelta(days=interval)` via `.isoformat()` → `2026-09-19T02:13:11.350086+00:00`); compared lexically at `:21593` (`get_due_flashcards`) and `:21622` (`count_due_flashcards`) against `CURRENT_TIMESTAMP` (`2026-09-19 02:13:11`, space separator). Same shape mismatch in `get_study_stats` `:23523,:23535,:23549` (`start_date.isoformat()` bound against `reviewed_at`/`updated_at`/`started_at` columns that are `DEFAULT CURRENT_TIMESTAMP`).
- Evidence: `$PY <SCRATCH>/repro_flashcard_next_review.py` → `raw next_review: ('text', '2026-09-19T02:13:11.350086+00:00', '2026-09-18 02:13:11')`. `$PY <SCRATCH>/repro_flashcard_due_day.py` (card set due **one hour ago** in the shape the writer produces, then the same instant in the column's own shape) →
  `T-form ... next_review<=now -> 0 ; count_due=0`
  `space  ... next_review<=now -> 1 ; count_due=1`
  `review 1h inside a 30-day window, stats(days=30)['reviews'] -> {'total_reviews': 0, 'avg_rating': None}` (control with the review well inside the window → `total_reviews: 1`). Mechanism: on the due date `'T'` (0x54) sorts after `' '` (0x20), so `next_review <= CURRENT_TIMESTAMP` is false until the date component advances.
- Why it matters: `Study_Interop/local_study_service.py:88,901` feed the Library rail badge (`count_due_flashcards`) and the next-card picker (`get_due_flashcards(limit=1)`); an "Again" card (interval=1) reviewed at 09:00 does not come back until 00:00 UTC two days later instead of 09:00 next day; the stats window silently excludes up to a day of reviews.
- Recommended correction: write `next_review` in the column's own shape (`next_review.strftime("%Y-%m-%d %H:%M:%S")`, or compute in SQL `strftime('%Y-%m-%d %H:%M:%S','now','+N days')`) and bind `start_date.strftime(...)` in `get_study_stats`; or compare through `julianday()` as `get_conversations_for_character` already does (`:11456-11471`). Existing rows carrying the `T…+00:00` shape need a one-shot `UPDATE ... SET next_review = strftime('%Y-%m-%d %H:%M:%S', next_review)` (SQLite's `strftime` parses the ISO form).
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/ChaChaNotesDB/test_study_functionality.py::test_get_due_flashcards` (:274–291) and `:457` exercise only `next_review IS NULL` / empty-deck cases; no test binds a reviewed card's due instant (grep of assertions, not executed).
- Already covered: none

### P2 [D1] — Two readers use the raw connection as a context manager, which commits any caller-owned transaction on exit
- Where: `ChaChaNotes_DB.py:12752` (`get_conversation_context_summary`) and `:12802` (`get_conversation_console_project_context`): `with self.get_connection() as conn:`. `sqlite3.Connection.__exit__` calls `commit()`; every sibling reader uses `with self.transaction() as conn:` which borrows a live transaction and never commits it (`:24018-24025`, `:24159`).
- Evidence: `$PY <SCRATCH>/repro_conn_ctx_commit.py` → control (inner read via `transaction()`, caller aborts) `title after abort: t`; candidate (inner read via `get_conversation_context_summary`, caller aborts) `in_transaction after inner read: False` … `title after abort: LEAKED`. The caller's `rollback()` becomes a silent no-op.
- Why it matters: any future caller that reads the summary/project-context inside an outer `transaction(immediate=True)` gets its partial writes committed and its rollback discarded with no error. Today's two production callers (`Chat/console_chat_store.py:22119` in `_resolve_context_summary_on_resume`, `Chat/chat_persistence_service.py:2603`) are not inside a transaction (checked: no `transaction(`/`BEGIN` between the enclosing `def` at `:22083` and the call), so this is a trap, not a live loss.
- Recommended correction: replace both `with self.get_connection() as conn:` with `with self.transaction() as conn:` (the file's own idiom; 2 lines).
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/UI/test_console_resume_active_path.py:1388,1407`, `Tests/Chat/test_console_chat_store_project_instructions.py:54,69` assert returned values only, not transaction ownership.
- Already covered: none

### P2 [D4] (b) — Six timestamp shapes are written to rows; two of them have already forced `julianday()` workarounds and the un-worked-around sites sort lexically
- Where (writers): (a) `_get_current_utc_timestamp_iso` `:8700` → `YYYY-MM-DDTHH:MM:SS.mmmZ` (≈45 call sites); (b) SQL `CURRENT_TIMESTAMP` → `YYYY-MM-DD HH:MM:SS`: every column DEFAULT (87 occurrences in the schema literals) plus 14 explicit Python-path writes `:13587 :21479 :21566 :21567 :21593 :21623 :21732 :21787 :21836 :21891 :21904 :22069 :22124 :23456` and `:3777`; (c) SQL `strftime('%Y-%m-%dT%H:%M:%fZ','now')` `:5773`, `:9337 :9342 :9396` and the `character_expression_images` defaults (same shape as (a) — correct, `%f` = SS.SSS); (d) `datetime('now')` in the `world_book_entries_sync_delete` trigger `:2013` (shape (b)); (e) `datetime.isoformat()` with microseconds + `+00:00` `:21570`, `:23523/:23535/:23549` (the P1 above); (f) read-side re-serialisation `:18728` `raw_timestamp.isoformat().replace("+00:00","Z")` → `…ffffffZ` (differs from (a)'s `mmmZ`); (g) `sqlite_datetime_fix.adapt_datetime` turns any `datetime` bind parameter into shape (e).
- Evidence: the file's own comment at `:17723-17734` ("`last_modified` is DATETIME DEFAULT CURRENT_TIMESTAMP, whose space-separated shape sorts against the ISO `T...Z` shape application writers stamp -- which is why the ACTIVE notes list wraps its date ordering in `julianday()` (task-32172)") and the `julianday()` wrappers at `:11456-11471`. Lexicographic `ORDER BY last_modified DESC` remains at `:11184 :11820 :11905 :17737 :18157 :18562 :19689 :19866 :22689`. `grep -rlE 'isoformat\(timespec="milliseconds"\)' tldw_chatbook` → 8 files / 13 copies of the (a) formatter (`Personal_Context/repository.py`, `Canvas/staging.py`, `Canvas/repository.py`, `Notes/note_import_executor.py`, `Notes/note_folder_repository.py`, `Notes/notes_organization_repository.py`, `Notes/Notes_Library.py`, `DB/ChaChaNotes_DB.py`); no `Utils` helper exists (`grep -rnE 'def (utc_now|now_utc|utc_iso|iso_now|utc_timestamp)'` → none). `grep -iE 'timestamp|datetime' adr_list.txt` → no ADR.
- Why it matters: rows written by a DEFAULT (or by any writer that omits the column) and rows written by (a) sort in the wrong order whenever they share a date; the flashcard/stats P1 is the same defect class reaching a comparison. That the (a)/(b) mix reaches `conversations.last_modified`/`notes.last_modified` in real profiles is **inferred** from the `:17723` comment and task-32172, not demonstrated here.
- Recommended correction: one canonical formatter in `Utils/` (e.g. `Utils/time_format.py::utc_now_iso()` returning shape (a)) adopted by the 13 copies; new columns default to `(STRFTIME('%Y-%m-%dT%H:%M:%fZ','NOW'))` as `character_expression_images` already does; date ORDER BYs over columns that can carry both shapes go through `julianday()` (the `:11456` precedent). Retiring the existing (b) DEFAULTs is a storage-format change → L.
- Size: M for the helper + writer adoption; L for the schema defaults · ADR: new (storage timestamp format) · Confidence: verified for the writer census, inferred for the mixed-row claim on conversations/notes
- Pinning test: none found for ordering across shapes.
- Already covered: task-32172 (Notes date ordering only)

### P2 [D4] (b) — Library FTS tokenisers re-roll `Utils.fts5_match_forms.build_and_match_query` with different tokenisation, a 20-token cap and no NUL guard
- Where: `ChaChaNotes_DB.py:17838` `_library_note_fts_query`, `:18471` `_library_conversation_fts_query` (verbatim clones), plus `DB/Prompts_DB.py:3330` `_library_prompt_fts_query`, `DB/Client_Media_DB_v2.py:8612` `_library_fts_query` (same shape; outside this slice). Canonical: `Utils/fts5_match_forms.py:348` `build_and_match_query` (used by this file's other 6 search seams).
- Evidence: drift script (inline in this review) comparing `CharactersRAGDB._library_note_fts_query(q)` vs `build_and_match_query(q)`:
  `'foo-bar'` → library `'"foo" "bar"'` | canonical `'"foo-bar"'` DIFF · `"it's"` → `'"it" "s"'` | `'"it\'s"'` DIFF · `'a\x00b'` → `'"a" "b"'` | `''` DIFF (canonical refuses NUL; library binds it and SQLite truncates the parameter) · 22 tokens → library truncates to 20 | canonical keeps all DIFF · `'hello world'` SAME.
- Why it matters: Library ▸ Notes/Conversations search answers a different question from Console/character/flashcard search for hyphenated or apostrophe-bearing terms (`"foo-bar"` is a phrase over the runs `foo bar`; `"foo" "bar"` is AND in any order) and silently drops tokens past 20.
- Recommended correction: make the four library copies call `build_and_match_query` (canonical home already exists); if the 20-token cap is wanted, add it as a parameter there. `Tests/Utils/test_fts5_quoting_adoption_census.py` guards the quote escape only, not the tokeniser.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/DB/test_fts5_quoting_search_seams.py` covers quoting on the canonical seams; none pins the library tokeniser's split-on-punctuation.
- Already covered: none

### P2 [D3] — God module: one class, 24,180 lines, 15 responsibility clusters
- Where: `CharactersRAGDB` `:709–23951` (+ module helpers `:1–708`, `TransactionContextManager` `:23952–24180`). Clusters with ranges: (1) module-level validators/authorizations/SQL splitters `:1–708`; (2) schema + 25 migration SQL literals as class attributes `:720–3305`; (3) connection lifecycle, quiescence, local-authority, backup/integrity `:3306–3988`; (4) `execute_query`/`execute_many`/`transaction` `:3989–4197` + the context manager `:23952–24180`; (5) migration runner primitives + 69 `_migrate_from_*` steps `:4198–8420`; (6) `_initialize_schema` + per-open repair hooks `:8421–8720`; (7) character cards + expression images + FTS `:8951–10402`; (8) conversations (identity normalisation, archive, search/paging/locator, cursor/summary/project-context, delete/restore) `:10403–13165`; (9) messages (adaptive insert, continuation/generation projection, attachments/generation metadata, update/tombstone, exchanges, trajectory, annotations, variants, sync-delete proofs, FTS) `:13166–16493`; (10) generic-item CRUD + keywords/collections `:16494–17461`; (11) notes + owner proofs + links + Library note seams + organization/receipts/dispatch `:17462–19431` (with the Library conversation seams interleaved at `:18466–18874`); (12) link tables + sync_log intents/retention/prune + `backfill_messages_fts` `:19432–21214`; (13) flashcards/decks/templates/assets `:21242–22256`; (14) quizzes/questions/attempts/grading incl. a Levenshtein implementation `:22257–23478`; (15) learning paths/topics/stats + kept briefings/scripts `:23402–23951`.
- Evidence: `grep -nE '^    def ' | wc -l` ≈ 430 methods; `awk` map at review start (line numbers above).
- Why it matters: any change pays a whole-file read/edit cost (this review needed 13 chunks); clusters 13–15 share nothing with 7–12 beyond the connection.
- Recommended correction: out of scope to redesign here — a split must follow `backlog/docs/library-decomposition-recipe.md` (§1 per-subsystem PR series, §2 field-ownership script, §17 file-size governance). The obvious first PRs are the self-contained study cluster (13–15, ~2.7k lines) and the migration steps (5, ~4.2k lines, 27 copies of one 45-line try/verify block — see the P3 below).
- Size: L · ADR: yes (`backlog/docs/library-decomposition-recipe.md` governs the shape; no ADR names this file) · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D3] — loguru and stdlib `logging` both used; one live stdlib call sits on the message-update write path
- Where: `import logging` `:47`; `logging.warning(...)` `:15006` in `_update_message_uncoordinated` (bypasses every loguru sink); `logging.ERROR` `:8613` is only the level constant passed to `persist_event` (legitimate).
- Evidence: `grep -nE '\blogging\.'` → exactly the two sites above.
- Why it matters: the unknown-field warning on message updates goes to the stdlib root logger, not the app log.
- Recommended correction: `logger.warning` at `:15006`; keep the `logging.ERROR` constant (or use loguru's level name).
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D3] — Dead per-call imports of already-imported stdlib names, and one sibling-inconsistent in-body import
- Where: `import re` `:16141` (`re` imported at `:41`); `from datetime import datetime, timedelta` `:21557` and `:23508` (`datetime` imported at `:45`; only `timedelta` is new); `from tldw_chatbook.Backup_Recovery.participants import _core_access` `:3441` inside `_get_thread_connection` while the same module's five siblings are imported at `:91-97` (no test patches `_core_access`: `grep -rn 'patch.*_core_access' Tests` → none).
- Evidence: greps above. The ~25 other in-body imports of `tldw_chatbook.Chat.*` / `Sync_Interop.*` are **not** in this finding: none of those modules is loaded after `import tldw_chatbook.DB.ChaChaNotes_DB` (checked via `sys.modules`), `Chat/__init__.py` eagerly imports `chat_conversation_service` etc., and `Chat/console_semantic_revision.py:33` imports `CharactersRAGDB` at module level — so they are cycle-avoidance, verified-fine.
- Recommended correction: hoist `timedelta` and `_core_access` to the module imports; delete the `import re`.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D4] (b) — 27 migration steps are the same 45-line try/execute/verify/except block; two steps re-roll helpers the file already has
- Where: `_migrate_from_v4_to_v5` … `_migrate_from_v25_to_v26`, `v28→v29`, `v18→v19`, `v7→v8`, `v29→v30`, `v30→v31`, `v31→v32`, `v41→v42` (`grep -c 'Unexpected error during migration'` → 27). `_migrate_from_v42_to_v43` `:6560-6574` re-implements `_split_sql_statements`/`_migration_file_statements` (`:501`, `:4350`) inline; `_migrate_from_v57_to_v58` `:7661-7669` is a verbatim copy of `_repair_missing_notes_organization_sync_ids` `:8635`.
- Evidence: reading; greps above.
- Why it matters: one-shot code, so no runtime cost — but ~1,200 lines of the god module are one template, and the two inline re-rolls are exactly the shape task-19553 removed elsewhere (the inline splitter in v42→v43 lacks the `ADD COLUMN`-skip and `DROP TRIGGER` idempotence `_execute_migration_statements` carries).
- Recommended correction: a table-driven `_run_sql_constant_step(conn, from_v, to_v, sql)` for the constant-backed steps; route v42→v43 through `_execute_migration_statements`; call the existing repair helper from v57→v58. README's rule is behavioural (one guarded transaction per step, re-enterable) and is preserved.
- Size: M · ADR: no (README governs) · Confidence: verified · Pinning test: `Tests/DB/test_chachanotes_bare_open_self_migration.py`, `Tests/Packaging/test_installed_distribution.py` pin the chain's behaviour (would stay green) · Already covered: none

### P3 [D4] (b) — Five identical LIKE-escape one-liners across four DB modules; no shared helper
- Where: `ChaChaNotes_DB.py:17833`, `:18466`, `DB/Client_Media_DB_v2.py:8607`, `DB/Prompts_DB.py:3325`, `DB/character_conversation_search.py:1417` — all `value.replace("\\","\\\\").replace("%","\\%").replace("_","\\_")`.
- Evidence: bodies printed side by side (identical; no drift). `grep -rnE 'def [a-z_]*escape[a-z_]*like' tldw_chatbook/Utils` → none.
- Recommended correction: `Utils/sql_like.py::escape_like(value)` (or next to `escape_identifier` in `DB/sql_validation.py`); 5 mechanical swaps.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D4] (b) — `ConflictError.__str__` is defined three times with an attribute-name drift
- Where: `ChaChaNotes_DB.py:386-419` (`entity_id`), `DB/Client_Media_DB_v2.py:114` and `DB/Prompts_DB.py:107` (`identifier`; those two are byte-identical to each other, md5 `2b3a0104…`).
- Evidence: md5 of the three bodies; Media body printed.
- Why it matters: callers that catch a `ConflictError` from one DB and format the other cannot rely on one attribute name.
- Recommended correction: one `ConflictError` in `DB/base_db.py` (or `DB/exceptions.py`) with a single attribute; the three modules re-export it.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D4] (b) — Paired Library keyword fetchers and the `IN (...)` chunking convention are applied inconsistently
- Where: `_library_keywords_for_notes` `:17854` / `_library_keywords_for_conversations` `:18487` (+ Prompts `:3343`, Media `:8628`) differ only in table/column names. Separately, 500-id chunking (SQLite host-parameter ceiling) is present in `:11056 :11286 :12244 :14320 :14408 :17640` and absent from the same-shaped `:12036 :12142 :14732 :16049 :16103 :17861 :18494 :19570 :19830 :23918`.
- Evidence: reading; every absent site is caller-bounded today (a page of ids), so no live failure.
- Recommended correction: one `_chunked_in(ids, size=500)` iterator used by all IN-list readers; one generic `_keywords_for(conn, link_table, id_col, ids)`.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D3] — `search_flashcards` has no production caller and no `LIMIT`
- Where: `:23479-23504`.
- Evidence: `grep -rnE 'search_flashcards\(' --include='*.py' .` → only `Tests/DB/test_fts5_quoting_search_seams.py` (:143,:144,:262,:612,:655); `list_flashcards(q=...)` `:22200` is the live search path and has `LIMIT`.
- Recommended correction: delete it (and its five test references) or add `limit` like `list_flashcards`.
- Size: S · ADR: no · Confidence: verified (grep incl. tests; no re-export found) · Pinning test: the seam test above states quoting behaviour, not that the method must exist · Already covered: none

### P3 [D1] — Message-variant methods collapse `ConflictError`/typed errors into a generic `CharactersRAGDBError`
- Where: `create_message_variant` `:16290`, `get_message_variants` `:16341`, `select_message_variant` `:16408` — `except Exception as e: raise CharactersRAGDBError(...)`. Only `InputError` is re-raised first; a `ConflictError` (a `CharactersRAGDBError` subclass) is re-wrapped and loses its type/entity.
- Evidence: reading (`:16289-16295`).
- Recommended correction: add `except CharactersRAGDBError: raise` before the generic clause (the pattern the rest of the file uses).
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D1] — `_deserialize_row_fields` logs the first 100 chars of a malformed JSON field's content
- Where: `:8830-8832` (`Value: '{item[field][:100]}...'` at WARNING). Fields are `alternate_greetings`/`tags`/`extensions` on character cards and quiz `options`/`tags_json`/`source_citations_json`/attempt `answers`.
- Evidence: reading.
- Why it matters: user content reaches the log on a corruption path; low sensitivity, but it bypasses the `content_fingerprint` discipline the rest of the file follows.
- Recommended correction: log `len(item[field])` or `content_fingerprint(item[field])`.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D4] (a) — `CharactersRAGDB` re-implements `BaseDB`'s path/`:memory:` handling instead of inheriting it
- Where: `:3343-3351` vs `base_db.py:772-783`; `check_integrity` `:3954` vs `base_db.py:851`; `vacuum` `:21215` vs `base_db.py:834`. Five other DB classes subclass `BaseDB` (`AgentRuns_DB`, `Library_Collections_DB`, `Library_Ingest_Jobs_DB`, `Workspace_DB`, `Subscriptions_DB`); `ChaChaNotes`, `Client_Media_DB_v2`, `Prompts_DB` do not.
- Evidence: `grep -rnE '^class \w+\(.*BaseDB\)' tldw_chatbook/DB`.
- Caveat: `BaseDB._get_connection`/`check_integrity`/`vacuum` open **unmanaged** raw connections (no WAL/FK pragma, no quiescence registration), so the ChaChaNotes versions are the correct ones for this class; only the path-handling block is a true re-roll. Report as duplication, not as "use BaseDB's connection".
- Size: S (path block) · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

## Candidate dispositions
| candidate (file:line pattern) | confirmed / retired (why) / unverified (check) |
|---|---|
| dup_shape: `get_conversation_active_leaf@:12707` ~ `add_keyword@Prompts:1271` ~ `_provider_config@settings_screen:12619` ~ `release@pending_handoff_store:445` | retired — mechanical 3-line wrapper shape; no shared logic |
| dup_shape: `_library_note_fts_query@:17838` / `_library_conversation_fts_query@:18471` (+Prompts/Media) | **confirmed** — P2 D4(b) with measured drift vs `build_and_match_query` |
| dup_shape: `_library_keywords_for_notes@:17854` / `_library_keywords_for_conversations@:18487` (+Prompts/Media) | confirmed — P3 D4(b), table/column-only drift |
| dup_shape: `ConflictError.__str__@:409` (+Media:114, Prompts:107) | confirmed — P3 D4(b), `entity_id` vs `identifier` drift |
| except_exception_pass `:8618` | retired — documented diagnostics guard inside the failure path; original error re-raised at `:8621-8623` |
| fetchall_dynamic_sql (46 rows: 9448 9588 9631 10354 11189 11316 11367 11477 11830 11923 11997 12049 12127 12161 12219 13063 13151 14690 14755 15715 16478 16746 17096 17814 17870 18503 18898 19424 19562 19582 19693 19740 19755 19804 19843 19870 19934 20266 20650 20775 21608 22253 22695 22960 23364 23504) | retired as injection candidates — every interpolated fragment is a class-internal literal, a whitelisted clause (`_CHARACTER_SORT_CLAUSES`, `_conversation_archive_scope_clause` dict, validated `ASC/DESC`), a `?` placeholder list, or a `validate_table_name`-checked identifier; all values are bound. Note: `_list_generic_items@:16746` and `_search_generic_items_fts@:17096` interpolate `order_by_col`/`fts_match_cols_or_table` unvalidated but only 2 literal callers exist (P3 hardening, not a finding). LIMIT status of each row is covered by the no-LIMIT classification below. |
| fetchall_no_limit — migration/one-shot rows (4292 5240 5408 5463 5511 5594 5679 5945 6022 6442 6461 6581 6599 7514 7552 7661 7679 7717 7754 7791 7864 7974 8041 8066 8106 8141 8182 8640) | retired — PRAGMA/`sqlite_master`/`foreign_key_check` results or NULL-`sync_id` rows inside the boot transaction; `:8640` also runs on every open for v≥58 but selects only NULL-`sync_id` rows (unique index covers NULLs) |
| fetchall_no_limit — chunked IN-lists (11058 12252 14328 14416 17647) | retired — 500-id chunks |
| fetchall_no_limit — COUNT/exact-row reads (17734 18162 18332 18567 18673 18841 21998 20102) | retired |
| fetchall_no_limit — per-parent-key sidecar reads (9325 15153 15467 15495 15778 15924 16049 16104 17905 17916 17977 18000 18039 18938 19230 19597 23924 23948) | retired — bounded by one conversation/message/note/page; `:15467` returns every `capture_blob` for one message (Inspector detail path, documented local-only) |
| fetchall_no_limit / dynamic without LIMIT — UI-adjacent (9631 `list_distinct_character_tags`; 11923 `locate_conversation_page`; 11997 `get_all_conversation_ids`; 18898 `get_all_note_ids`; 12161 `get_messages_for_conversation_by_parent_ids`; 12219 `get_message_tree_rows_for_conversation`; 14755 batch; 15715 annotations; 19562/19582/19740/19804/19843 keyword joins; 20650 intents) | retired — bounded by tag cardinality / one page window (ROW_NUMBER CTE) / documented export scope (ids only) / one conversation (no BLOB at 12219; 12161 still selects `image_data` for the given parents) / `ROW_NUMBER() <= limit` / one conversation / link count. None is an unbounded UI list. |
| fetchall — `get_sync_log_entries@:20775` `limit=None` default | unverified — bounded only when the sync caller passes `limit`; check `grep -rn 'get_sync_log_entries(' tldw_chatbook` for a call omitting it |
| fetchall — `search_flashcards@:23504` | confirmed — P3 dead public method, no LIMIT |
| function_body_import_per_file (36) | 4 confirmed (P3: 3441, 16141, 21557, 23508); the rest retired — cycle avoidance through `Chat/__init__.py` (evidence in the P3 finding) |
| legacy_markers_per_file (22) | not examined — marker semantics not defined in the excerpt; comments only |
| loguru_and_logging `:0` | confirmed — P3 (`:15006` live call; `:8613` level constant only) |
| seed_name `_initialize_schema@:8421` | examined, verified-fine — migration table keys 4…72 contiguous (`:8502-8571`), one `BEGIN IMMEDIATE` for the whole chain, up-to-date path runs three cheap repair hooks per open (`:8688-8691`) |
| strftime `:5773` `%Y-%m-%dT%H:%M:%fZ` | retired — `%f` = SS.SSS in SQLite, correct; same shape as the Python formatter |

## Verified-fine
- **base_db bypass (item 2):** no `sqlite3.connect`; the single opener is `connect_private_sqlite(... factory=_QuiescentSQLiteConnection)` `:3479-3487`, backups via `backup_connection_to_private` `:3919`. `BaseDB._get_connection` is never used by this class (and must not be — it is unmanaged).
- **`str(params)` logging (item 5):** both query-logging sites are `logger.opt(lazy=True)` + `preview_params` (`:4030-4038`, `:9847-9853`); `execute_many` logs only `len(params_list)`; error paths log `query[:300]` (SQL text, no values). Retired.
- **`except Exception` on write paths (item 4):** every generic clause re-raises as `CharactersRAGDBError` or (backup/integrity `:3945 :3980`) returns `False` after logging; none swallows. Only type-loss remains (P3 above).
- **Migration hygiene (item 7):** `_CURRENT_SCHEMA_VERSION = 73` `:735`; steps dict contiguous 4→72; every step guarded by `_require_migration_entry_version` or an inline version check and a rowcount-checked bump; `executescript` absent from the schema path (task-19553). VALID_TABLES/index census: preflight passed, not re-derived.
- **Quiescence tax (item 8 — task-31502, cite only):** confirmed at `ChaChaNotes_DB.py:3444-3447/3573` (`begin_acquisition`/`finish_acquisition` on every `get_connection()`), `:3480` (`_QuiescentSQLiteConnection` factory), `:3486` (`attach_quiescence_registry`), `:23983/:24055` (`begin_use`/`end_use` per `transaction()`), and `base_db.py:362-376` + `:463-473` (`_QuiescentSQLiteCursor.execute` → `registry.begin_use()` per statement), `:524-530` (`fetchall` release), `:195-216` (`begin_use`/`end_use` take the shared `threading.Condition(RLock)` and `end_use` calls `notify_all()`), `:350-376` (per-connection `_quiescence_tokens_lock`). Net: ≥4 lock acquisitions + 1–2 `notify_all` per statement, plus `_core_access`/`_core_operation` (Backup_Recovery) per call. Already covered by task-31502.
- **`get_conversations_for_character` `ORDER BY julianday(last_modified)`** `:11456-11471`: defeats the index but is the correct workaround for mixed shapes and is bounded by one character's conversations.
- **Chat/Sync in-body imports (~25 sites):** cycle avoidance (see P3 imports finding for the evidence).
- **`get_message_tree_rows_for_conversation`** `:12163`: no LIMIT by design (TASK-22206), no BLOB, `idx_msgs_conv_ts` plan asserted by its docstring.
- **`_enrich_default_assistant_card_if_bare`** FTS `rebuild` + conditional UPDATE `:4624`: shared by fresh install and v31→v32; documented (task-2451).
- **`add_message` adaptive column list** `:13183`: only drops a column when its value is `None`; raises otherwise (README rule).
- **Console `documented duplication`** note from the brief is out of this slice.

## Retired
- "`fetchall()` without LIMIT on messages/conversations/notes is a UI-list hazard": 42 non-migration methods examined one by one (table above); none feeds an unbounded UI list — each is page-windowed, id-chunked, per-parent-key, or a documented export/ids-only scope. Symptom (many `fetchall()`s) real, cause wrong.
- "`str(params)` reaches logs on hot queries": both sites already lazy + redacted (`:4030`, `:9847`).
- "Function-body imports are dead deferrals": true for 4 sites only; the ~25 `Chat.*`/`Sync_Interop.*` sites are cycle avoidance (`sys.modules` check + `Chat/__init__.py` eager imports + `console_semantic_revision.py:33` importing `CharactersRAGDB`).
- "`BaseDB` helpers ignored" for connection/integrity/vacuum: `BaseDB`'s versions open unmanaged connections; ChaChaNotes' overrides are the safe ones. Only the path block is a genuine re-roll (P3).
- First flashcard repro attempt (`repro_flashcard_next_review.py`) showed `count_due=0` after an "Again" review — that alone is correct (due tomorrow); the defect only shows on the due day, which `repro_flashcard_due_day.py` isolates.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| `conversations.last_modified` / `notes.last_modified` actually carry both timestamp shapes in real profiles (basis of the lexicographic-ORDER-BY half of the P2 timestamp finding) | needs a real profile DB; only the file's own `:17723` comment + task-32172 are cited | `cd <worktree> && source <SCRATCH>/env.sh && $PY -c "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(c.execute(\"SELECT substr(last_modified,11,1) sep, COUNT(*) FROM notes GROUP BY sep UNION ALL SELECT substr(last_modified,11,1), COUNT(*) FROM conversations GROUP BY 1\").fetchall())" <path-to-a-copied-user-ChaChaNotes.db>` |
| `get_sync_log_entries` is ever called with `limit=None` (unbounded) | callers not traced | `grep -rnE 'get_sync_log_entries\(' --include='*.py' tldw_chatbook \| grep -v 'limit='` |
| `execute_query`'s per-call `log_histogram`/`log_counter` (`:4056-4066`) cost on query-heavy paths | not measured; `Metrics/metrics_logger` not read | `cd <worktree> && source <SCRATCH>/env.sh && $PY -c "import timeit; from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB as C; db=C(':memory:','t'); print(timeit.timeit(lambda: db.execute_query('SELECT 1').fetchone(), number=2000)/2000*1e6, 'us/query')"` |
| `legacy_markers_per_file` excerpt row (22 markers) | marker definition not in the excerpt | `grep -niE 'legacy' tldw_chatbook/DB/ChaChaNotes_DB.py \| wc -l` and read each |
