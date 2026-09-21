# DB-rest — tldw_chatbook/DB/ minus ChaChaNotes_DB / Client_Media_DB_v2 / base_db / sql_validation, 32 files, 34,245 lines

COMPLETE. (Written incrementally; the first run was killed by a usage limit after `Subscriptions_DB.py` and one finding. This run finished every remaining file. Finding bodies appear in review order; the ordered index under `## Findings` is the required P0→P3 / D1→D4 ordering.)

**16 findings: 0 P0 · 2 P1 · 8 P2 · 6 P3.**

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| Subscriptions_DB.py | 6641 | read in full (1-6641, 14 sequential chunks) |
| Prompts_DB.py | 5128 | read in full (1-5128, 9 sequential chunks) |
| private_sqlite.py | 4019 | read in full (1-4019; the 555-line `_SQLITE_OWNER_POLICIES` registry at 116-670 skimmed as data) |
| AgentRuns_DB.py | 3350 | read in full (1-3350, 6 chunks) |
| Evals_DB.py | 2477 | read in full (1-2477, 5 chunks) |
| character_conversation_search.py | 1666 | read in full (1-1666, 4 chunks) |
| recovery_operations.py | 1305 | code read in full (1-142, 142-330, 954-1305); the ~800 lines of embedded DDL-string catalogs (`_WORKSPACES_SCHEMA`, `_AGENT_RUNS_SCHEMA`, `_SUBSCRIPTIONS_SCHEMA`) read as data, not line by line |
| automatic_work.py | 846 | sampled: 1-60 (module contract + `transaction()`), 140-215 (`_check_admission`/`_start_automatic`), plus the whole-file AST DML/handler scans |
| RAG_Indexing_DB.py | 846 | read in full (1-430 line by line; 430-846 scanned by symbol + the two datetime call sites read in full) |
| Workspace_DB.py | 836 | sampled: 1-120 (class + v3 migration SQL), 470-836 (`close`/`_initialize_schema`/migrations/`get_schema_version`); the 350 lines of v4-v8 migration DDL strings read as data |
| Library_Collections_DB.py | 806 | sampled: 480-720 (connection/transaction/close/_initialize_schema) + whole-file AST scans |
| VisualIdentity_DB.py | 800 | sampled 1-120 + candidate rows (`:47`, `:165`); no own connection — every query goes through `CharactersRAGDB` |
| recovery_core_schema.py | 747 | mechanical only (embedded installed-schema DDL catalogs; no executable logic beyond the tuples) |
| Chunking_Lab_DB.py | 654 | sampled: 30-120 (`_connection`/schema), 400-654 (`save`/`_replace_transaction`/`_collect`/`clear`/`close`) + AST scans |
| private_sqlite_process.py | 619 | mechanical only (referenced from `private_sqlite` reading; not read line by line) |
| private_sqlite_files.py | 543 | read 1-130 + full AST diff of all 9 functions vs private_sqlite.py |
| Library_Ingest_Jobs_DB.py | 501 | mechanical (symbol list + AST DML/handler scans + the `all_jobs`/`_upsert_job`/`delete_job` sites) |
| recovery_core.py | 478 | sampled: `relocate` (`:276`) diffed against `recovery_operations.py:106`; rest mechanical |
| private_sqlite_protocol.py | 433 | mechanical only |
| agent_worktrees.py | 306 | mechanical (symbol list + candidate rows `:31 _now`, `:68 _identity`, `:211 list_for_conversation`) |
| private_sqlite_helper.py | 306 | sampled `:272` (`files.prepare_batch`, the POSIX prepare path) |
| chachanotes_fts_backfill.py | 224 | mechanical (symbol list; pacing primitives read in `fts_backfill_pacing.py`) |
| sql_logging.py | 101 | read in full |
| transaction_observer.py | 98 | read in full |
| sql_identifier_core.py | 88 | read in full |
| recovery_sqlite.py | 87 | read in full |
| Workflows_DB.py | 83 | read in full |
| sqlite_datetime_fix.py | 79 | read in full |
| fts_backfill_pacing.py | 78 | read in full |
| private_sqlite_helper_entry.py | 56 | read in full |
| canvas_payload_validation.py | 42 | read in full |
| __init__.py | 2 | read in full |
| migrations/ (78 files) | — | mechanical: derived-artifact compliance spot-checked on the four newest `chachanotes_*` migrations (see Verified-fine) |

## Findings

**Ordered index (P0→P3, then D1→D4).** The bodies below are in review order; this index is the required ordering.

| # | sev | axis | finding | where |
|---|---|---|---|---|
| 1 | P1 | D2 | `search_prompts` materialises every matching id and binds it — linear cost, hard failure past 32766 matches | `Prompts_DB.py:3884-3939` |
| 2 | P1 | D2 | `reconcile_orphaned_runs` re-scans all run history with a per-run step query, from a `compose()` | `AgentRuns_DB.py:2340-2519` |
| 3 | P2 | D1 | `update_keywords_for_prompt` runs DML on the legacy-isolation held connection with no transaction | `Prompts_DB.py:1760-1870` |
| 4 | P2 | D1 | Markdown prompt export collapses distinct names to one `.md`, duplicate ZIP entries, still reports success | `Prompts_DB.py:5091-5100` |
| 5 | P2 | D1 | Three prompt-export paths write prompt bodies to a predictable 0644 path in the shared temp dir | `Prompts_DB.py:4806/:4971/:5008` |
| 6 | P2 | D3 | `_METADATA_COLUMNS` no longer matches its own "every column except steps" contract | `AgentRuns_DB.py:1671-1682` |
| 7 | P2 | D4a | `_loads_json_or_default` adopted at 4 of 15 JSON-column reads; 3 nullable columns still raw | `Evals_DB.py:1457-1476` |
| 8 | P2 | D4b | LIKE-metacharacter escape re-rolled 9× with no shared helper, one copy using a different escape char | 6 DB/Library modules + TTS |
| 9 | P2 | D4b | The private-SQLite artifact validator exists twice (≈440 lines), already drifted; the promised delegation never happens | `private_sqlite.py:765-1246` vs `private_sqlite_files.py:31-503` |
| 10 | P2 | D4b | Store-template drift: `except Exception` around 18 BEGIN/ROLLBACK guards wedges the held connection | 8 DB modules |
| 11 | P3 | D1 | `soft_delete_keyword` raises `ConflictError` with the table name as the message | `Prompts_DB.py:2838` |
| 12 | P3 | D1 | `search_tasks`' LIKE branch leaves `%`/`_` unescaped — and that branch is chosen *because* of them | `Evals_DB.py:1150-1166` |
| 13 | P3 | D1 | Unavailable-characters filter folds case in Python and in SQLite, so non-ASCII terms match nothing | `character_conversation_search.py:1378/:1526` |
| 14 | P3 | D1 | `needs_reindexing` rejects the naive `datetime` its own writer accepts | `RAG_Indexing_DB.py:827-828` |
| 15 | P3 | D4b | Three divergent filename sanitisers; the *shared* one keeps NUL and control characters | `Utils/text.py:47`, `Utils/file_extraction.py:614`, `Prompts_DB.py:5091` |
| 16 | P3 | D4b | `EvalsDB` writes two incompatible timestamp shapes into the same `updated_at` columns | `Evals_DB.py` (28 SQL + 7 Python sites) |

### Bodies (review order)

### P2 [D1] — `PromptsDatabase.update_keywords_for_prompt` executes DML on the legacy-isolation held connection with no transaction/commit; a bare caller's link changes are silently discarded at `close()` (same shape as DB-media-base's `MediaDatabase` P0, but no shipped caller reaches it bare)
- Where: `tldw_chatbook/DB/Prompts_DB.py:1760-1870` (method; comment at 1764 says "called within an existing transaction (e.g. from add_prompt)… don't start a new transaction here"); DML at `:1819` (DELETE links), `:1840` (INSERT OR IGNORE links), plus `_log_sync_event` INSERTs at `:1829/:1848`. Bare public wrapper: `tldw_chatbook/Prompt_Management/Prompts_Interop.py:230-233`. In-transaction callers (fine): `Prompts_DB.py:1679` (add_prompt) and `:2059` (update_prompt_by_id), both inside `with self.transaction()`.
- Evidence (reproduced, isolated env, temp-file DB):
  `PromptsDatabase(p).add_prompt(... keywords=["k1"]); db.update_keywords_for_prompt(pid, ["k2","k3"]); conn.in_transaction; db.close(); reopen; SELECT links` →
  `in_transaction after bare update_keywords_for_prompt: True` / `links seen on same conn: 2` / `links after close+reopen: ['k1']` / `keywords table rows: ['k1','k2','k3']` / `sync_log rows: 5`.
  I.e. the k2/k3 keyword ROWS persist (each `_add_keyword_full` runs its own `with self.transaction()`), but the link DELETE/INSERTs and their `sync_log` unlink/link events sit in an implicit legacy-isolation BEGIN that `close_connection()` (`Prompts_DB.py:504-517`, plain `conn.close()`) rolls back. Worse than pure loss: the store is left half-applied (new keyword rows, old membership).
  Reachability: `rg -n "update_keywords_for_prompt" tldw_chatbook Tests` → only `Prompts_Interop.py:230-233` (wrapper, itself uncalled anywhere in `tldw_chatbook/`) and 4 tests that read back on the SAME connection. No shipped path calls it bare today — hence P2, not P0.
- Why it matters: `Prompts_DB.transaction()` (`:659-693`) BORROWS when `conn.in_transaction` is already true (`in_outer=True` → no commit), so once a bare call has opened the implicit transaction, every subsequent `with db.transaction()` on that thread also stops committing — the whole thread's later prompt writes ride on the one uncommitted transaction until something calls `execute_query(commit=True)` or the app exits (then all lost). Any future UI use of the Interop wrapper (the obvious "edit keywords" affordance) turns this into a P0.
- Recommended correction: wrap the body in `with self.transaction() as conn:` (borrow semantics make the in-transaction callers a no-op change) — S. Also make `close_connection()` commit-or-warn if `conn.in_transaction` (the task-22224 flip is the real fix; this store is a documented EXCEPTION at `:404-413`).
- Size: S · ADR: no (task-22224 store-template rule; this file's exception is documented) · Confidence: verified
- Pinning test: `Tests/Prompts_DB/test_prompts_db_legacy.py::test_update_keywords_for_prompt_with_empty_list_removes_all` (:1133-1141) and `::test_update_keywords_for_prompt_is_idempotent` — both call it bare and read back on the same connection, so they PASS while the data is uncommitted; none states durability. No test closes+reopens.
- Already covered: none (task-22224 is the template rule; no task for this store's flip — the docstring says "its own task" but none exists in `backlog/tasks/` — `rg -l "Prompts_DB" backlog/tasks | rg 22224` → checked below in dispositions)

### PRIORITY SWEEP RESULT — "DML on a legacy-isolation held connection with no transaction/commit" across the rest of the slice: only `Prompts_DB` has it
- Method: AST scan (`<SCRATCH>/dml_scan.py`) over every `*.execute/executemany/executescript` whose first-arg SQL literal starts with INSERT/UPDATE/DELETE/REPLACE/CREATE/DROP/ALTER, reporting the lexically enclosing `with` items.
  `cd $WT && $PY <SCRATCH>/dml_scan.py tldw_chatbook/DB/Evals_DB.py` → 21 OK / 55 BARE, of which every BARE is DDL inside `_create_schema`/`_migrate_schema` **plus** the 4 FTS rebuild statements at `Evals_DB.py:639-645`; all of those are reached only from `_init_schema` (`:243-263`), whose body is `with self.connection() as conn: ... with conn:`. So Evals_DB's documented task-22224 EXCEPTION docstring (`:196-215`, "every write path relies on `with conn:`") is accurate — **verified-fine, no lost-write shape.**
- The other stores named in the sweep are *not* on legacy isolation at all: `Workspace_DB.py:383`, `Library_Collections_DB.py:529`, `AgentRuns_DB.py:332`, `RAG_Indexing_DB.py:144`, `Library_Ingest_Jobs_DB.py:102`, `Chunking_Lab_DB.py:96`, `Workflows_DB.py:27` all set `isolation_level = None` (true autocommit) on the held connection, so a bare DML commits at statement end and nothing can be lost at `close()`. `rg -n "isolation_level" tldw_chatbook/DB/*.py` is the evidence.
- The remaining BARE non-DDL rows are all helpers that take `conn` as a parameter and are only called from inside an enclosing transaction — checked individually: `AgentRuns_DB._publish_console_activity_in_transaction` (:1003/:1008, name states it), `AgentRuns_DB.insert_recovery_diagnostic` (:2446), `Library_Ingest_Jobs_DB._upsert_job`/`delete_job`, `Chunking_Lab_DB.save/_replace_transaction/_collect/clear`, `automatic_work._check_admission`/`_start_automatic` (both take `conn` from `AutomaticWorkLedger.transaction()`), `character_conversation_search._upsert_document` (`:1326`, a `@staticmethod` taking `connection`; all 3 callers — `:1197`, `:1273`, `:1315` — are inside `self._database.transaction(immediate=True)`). Autocommit stores make the question moot anyway.
- `scheduled_tasks_db` is `tldw_chatbook/Scheduling/db/` — outside this slice's paths; not scanned here.

### P2 [D4b] — the SQL-LIKE metacharacter escape is re-rolled 9 times with no shared helper, and one copy uses a different escape character
- Where (all copies, `rg -n 'replace\("%", "\\\\%"\)|replace\("%", "!%"\)' --type py tldw_chatbook`):
  - named `@staticmethod` helpers, byte-identical body + byte-identical docstring: `DB/Prompts_DB.py:3325-3327` `_escape_library_prompt_like`, `DB/Client_Media_DB_v2.py:8607-8609` `_escape_library_like`, `DB/ChaChaNotes_DB.py:17833-17835` `_escape_library_note_like`, `DB/ChaChaNotes_DB.py:18466-18468` `_escape_library_conversation_like`, `DB/character_conversation_search.py:1417-1420` `_escape_like_query`, `Library/library_collections_service.py:672-674` `_escape_collection_like`
  - inline, no helper: `DB/Subscriptions_DB.py:3146` and `:3236`
  - a long-form copy: `Web_Scraping/cookie_scraping/cookie_cloner.py:83-92` `escape_sql_like_pattern`
  - **the drifted one**: `TTS/profile_repository.py:1197-1198` `_escape_like_literal` escapes with `!` (`value.replace("!", "!!").replace("%", "!%").replace("_", "!_")`), i.e. a different ESCAPE character, so the two families are not interchangeable and a copy-paste between them silently stops escaping.
- Evidence: the grep above (10 hits, 9 of them the escape itself); `rg -n "LIKE" tldw_chatbook/DB/sql_validation.py tldw_chatbook/Utils/*.py` → no output, i.e. **no shared helper exists** — this is the D4(b) sub-case, not "helper ignored".
- Why it matters: nine copies of a security-adjacent string transform that must stay in lockstep with its `ESCAPE '\'` clause; the `!` variant proves the drift is already real, and the next re-roll is the one that forgets `replace("\\", "\\\\")` (escaping `%`/`_` while leaving the escape char unescaped lets a literal `\` in user text swallow the following character).
- Recommended correction: one `escape_like(value: str, escape: str = "\\") -> str` in `DB/sql_validation.py` (already the home for SQL-identifier safety and already imported by every DB module here), the 9 sites call it, the TTS caller passes `escape="!"`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found that states the duplication as a requirement.
- Already covered: none

### P3 [D1] — `PromptsDatabase.soft_delete_keyword` raises `ConflictError` with the table name as the message and the row id as `entity`
- Where: `tldw_chatbook/DB/Prompts_DB.py:2838` — `raise ConflictError("PromptKeywordsTable", kw_id)`
- Evidence: read only. `ConflictError.__init__` is `(self, message="Conflict detected...", entity=None, identifier=None, *, code="conflict")` (`Prompts_DB.py:90-98`) and `__str__` renders `f"{base} ({', '.join(details)})"` with `details` built from `self.entity`/`self.identifier` (`:101-108`). So this raise produces `"PromptKeywordsTable (Entity: <kw_id>)"` — no sentence, no identifier. Every other `ConflictError` raise in the file passes `(message, entity, identifier)` in order (e.g. `:1585`, `:2024`).
- Why it matters: the optimistic-lock failure a user hits when two windows edit the same prompt keyword surfaces as the bare string "PromptKeywordsTable"; it also breaks any caller matching on `.entity`.
- Recommended correction: `raise ConflictError("Failed to soft-delete keyword due to a version mismatch.", "PromptKeywordsTable", kw_id)`.
- Size: S · ADR: no · Confidence: verified — `cd $WT && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY -c "from tldw_chatbook.DB.Prompts_DB import ConflictError; e=ConflictError('PromptKeywordsTable',7); print(repr(str(e)),'entity=',e.entity,'identifier=',e.identifier)"` → `'PromptKeywordsTable (Entity: 7)' entity= 7 identifier= None`
- Pinning test: none
- Already covered: none

### P1 [D2] — `PromptsDatabase.search_prompts` materialises EVERY matching prompt id in Python and binds them as one `IN (?,?,…)` list: search cost grows linearly with the match count and the call hard-fails with `too many SQL variables` past 32766 matches
- Where: `tldw_chatbook/DB/Prompts_DB.py:3884-3922` (the two unbounded `fetchall()`s into `matching_prompt_ids`, then `:3919-3921` `conditions.append(f"p.id IN ({id_placeholders})")`), consumed by the count at `:3935` and the page at `:3939`.
- Evidence (isolated env, temp-file DB, rows inserted directly + FTS populated so the state matches a normal library):
  - `SQLITE_LIMIT_VARIABLE_NUMBER` on this build = **32766** (`$PY -c "import sqlite3; print(sqlite3.connect(':memory:').getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))"`).
  - 32800 prompts all matching `dragon` → `search_prompts("dragon", page=1, results_per_page=20)` → `DatabaseError: Failed to search prompts: Query execution failed: too many SQL variables`, raised from `Prompts_DB.py:3935` (`SELECT COUNT(p.id) FROM Prompts p WHERE p.deleted = 0 AND p.id IN (?,?,…`).
  - Cost below the cliff, same script at three sizes: `N=  2000 … elapsed=6.2 ms` / `N= 10000 … elapsed=29.3 ms` / `N= 32000 … elapsed=93.3 ms` — and only 20 rows are ever displayed. The work is linear in the match count, not the page size.
- Why it matters: this is the shipped prompt-search path (`UI/Console_Modules/prompts.py:1353` → `Prompts_Interop.py:475`, `Prompt_Management/prompt_scope_service.py:581`, `Library/library_local_rag_search_service.py:789`). A broad one-token query against a bulk-imported prompt library pays ~3 ms per 1000 matches on every search and stops working entirely — an error toast, not an empty result — once the library passes ~32.7k matching rows.
- Recommended correction: keep the id set in SQLite instead of round-tripping it — replace the `IN (...)` with `p.id IN (SELECT rowid FROM prompts_fts WHERE prompts_fts MATCH ?) OR p.id IN (SELECT prompt_id FROM PromptKeywordLinks WHERE keyword_id IN (SELECT rowid FROM prompt_keywords_fts WHERE prompt_keywords_fts MATCH ?))`, which is the exact shape `search_library_prompts_page` (`:3510-3519`) already uses in this same file. That makes both the count and the page O(page) and removes the parameter cliff.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none — `rg -n "too many SQL variables" Tests` → no hits; no test states the `IN`-list construction as a requirement.
- Already covered: none

### P2 [D1] — the Markdown prompt export collapses distinct prompt names to the same `.md` filename, writes duplicate entries into the ZIP, and still reports "Successfully exported N prompts"
- Where: `tldw_chatbook/DB/Prompts_DB.py:5091-5097` — `safe_filename = re.sub(r"[^\w\-_ \.]", "_", p_data["name"]) + ".md"`, then `open(os.path.join(temp_zip_dir, safe_filename), "w")` and `zipf.write(..., arcname=safe_filename)` inside the per-prompt loop; status line at `:5100`.
- Evidence (isolated env, temp-file DB, two prompts named `a:b` and `a?b`):
  `export_prompts_formatted(db, export_format="markdown")` → `Successfully exported 2 prompts to Markdown in a ZIP file.` / `warnings: ["Duplicate name: 'a_b.md'"]` / `namelist: ['a_b.md', 'a_b.md']` / `z.read("a_b.md") -> b'# a?b (f3ce4fda-032a'` — i.e. both prompts are in the archive under one name, the on-disk staging file was overwritten, and every normal extractor (`unzip`, Finder, `ZipFile.extractall`, `ZipFile.read`) yields only the last one. Prompt names are free text and `:` `?` `/` `|` `#` `,` all map to the same `_`, so a two-prompt collision needs no contrivance.
- Why it matters: a user exporting their prompt library silently gets fewer files than prompts, with a success message that says otherwise. Export is the backup path.
- Recommended correction: de-duplicate `arcname` per archive (append `-{p_data['id']}` or a `(2)` suffix on collision) and drop the on-disk staging entirely — `zipf.writestr(arcname, md_content)` removes both the temp directory and the overwrite.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (`rg -n "export_prompts_formatted" Tests` — see dispositions)
- Already covered: none

### P2 [D1] — the three prompt-export paths write user prompt bodies to a predictable, world-readable path in the shared temp directory
- Where: `tldw_chatbook/DB/Prompts_DB.py:4806-4810` (`prompt_keywords_export_{timestamp}.csv`), `:4971-4973` (`prompts_export_{timestamp}.csv`), `:5008-5010` (`prompts_export_markdown_{timestamp}.zip`) — each is `os.path.join(tempfile.gettempdir(), f"...{datetime.now().strftime('%Y%m%d_%H%M%S')}...")` followed by a plain `open(..., "w")` / `zipfile.ZipFile(..., "w")`. (The fourth `tempfile` use, `tempfile.mkdtemp()` at `:5007`, is correct — 0700 — and is only the staging dir.)
- Evidence (isolated env): the same script above printed `CSV path: /var/.../T/prompts_export_20260918_074546.csv mode: 0o644` and `KW CSV path: /var/.../T/prompt_keywords_export_20260918_074546.csv mode: 0o644`. The CSV contains `System Prompt` and `User Prompt` columns in full (`:4979-4992`).
- Why it matters: two defects at once. (a) The name is derived only from a one-second-resolution timestamp, so on Linux — where `gettempdir()` is the shared, world-writable `/tmp`, unlike macOS's per-user `/var/folders/.../T` — any local user can pre-create the path as a symlink and redirect the write (CWE-377), or simply read the 0644 result (CWE-378): prompt bodies are user content and often carry credentials/system instructions. (b) Two exports inside the same second overwrite each other.
- Recommended correction: `tempfile.mkstemp(prefix="prompts_export_", suffix=".csv")` (0600, unique, O_EXCL) at all three sites, returning the fd/path — same line count, no new dependency.
- Size: S · ADR: no (no `backlog/decisions/` file covers temp-file creation; grep of `<SCRATCH>/adr_list.txt` for "temp" is in dispositions) · Confidence: verified (path + mode observed; the Linux `/tmp` half is platform reasoning, not run here)
- Pinning test: none
- Already covered: none

### P3 [D4b] — three divergent filename sanitisers, and the *shared* one is the unsafe one (keeps NUL and control characters)
- Where: `Utils/text.py:47-59` `sanitize_filename` (denylist `[<>:"/\\|?*]` only — 3 importers: `Chatbooks/chatbook_creator.py:56`, `Local_Ingestion/audio_processing.py:41`, itself); `Utils/file_extraction.py:614-636` `_sanitize_filename` (denylist `<>:"|?*\/\n\r\t`, keeps NUL); `DB/Prompts_DB.py:5091` inline `re.sub(r"[^\w\-_ \.]", "_", …)` (allowlist).
- Evidence: `PYTHONPATH=$WT $PY -c` on input `'ev\x00il\x07/..\\name'` → `Utils.text -> 'ev\x00il\x07..name'` (NUL and BEL survive, and the `/`-removal *joins* `il` to `..`), `Prompts_DB inline -> 'ev_il__.._name.md'`.
- Why it matters: the module named as the shared helper is the weakest of the three; a NUL in a filename truncates the path at the OS boundary and a bare `..` component is produced by *removal* rather than replacement. Any future "just use the shared helper" cleanup of the Prompts_DB site would be a regression.
- Recommended correction: make `Utils/text.sanitize_filename` the allowlist form (`re.sub(r"[^\w\-. ]", "_", name)` + strip leading dots + length cap), then point `file_extraction` and `Prompts_DB` at it. Fix the helper before consolidating onto it.
- Size: M (3 call sites, one behaviour change) · ADR: no · Confidence: verified
- Pinning test: none found asserting NUL survives.
- Already covered: none

### P2 [D4b] — the private-SQLite artifact validator (≈440 lines of TOCTOU-hardened `openat`/`fstat` checks) exists as two copies that have already drifted, and the "delegation" the docstring promises never happens
- Where: `tldw_chatbook/DB/private_sqlite.py:765-1246` and `tldw_chatbook/DB/private_sqlite_files.py:31-503` — 9 same-named module-level functions: `_failure`, `_open_artifact_fd`, `_artifact_postcondition_holds`, `_path_error_from_oserror`, `_optional_sidecar_restart_or_absent`, `_prepare_posix_artifact_generation` (285L / 273L), `_prepare_posix_artifact`, `_prepare_windows_artifact`, `_prepare_artifact`. `private_sqlite_files.py`'s module docstring says *"The local seam temporarily delegates here during the staged migration."*
- Evidence:
  - AST comparison of every shared function (`ast.unparse` bodies): **4 identical** (`_failure`, `_artifact_postcondition_holds`, `_path_error_from_oserror`, `_optional_sidecar_restart_or_absent`) and **5 already divergent** — `_open_artifact_fd` (13 changed lines: the `private_sqlite` copy carries a `_NativeOpenOutcome` probe the files copy lacks), `_prepare_posix_artifact_generation` (42 changed lines: `_pin_job` preflight seam + `preflight_body_errors` bookkeeping vs `open_artifact_fd`/`postcondition_holds`/`identity_out` injection seams), `_prepare_posix_artifact`/`_prepare_artifact` (signature plumbing), `_prepare_windows_artifact` (`os.stat(selected, follow_symlinks=False)` vs `selected.lstat()`). The *validation predicates* are still equal today; only the seams differ.
  - The claimed delegation does not exist: `ast` scan of `private_sqlite.py` → `Name-node uses of 'private_sqlite_files': 0`, `raw substring count: 1` (the import at `:22` itself). **`private_sqlite.py:22` is a dead import.**
  - The in-module copy is **Windows-only**, hence dead on every platform this app actually ships on: `_prepare_artifact` is called at `private_sqlite.py:1866/:1872` (inside `if private_paths._WINDOWS_PLATFORM:` at `:1864`; the `else` is `prepare_in_helper`) and at `:2409/:2417` inside `_prepare_source_artifacts`, whose only caller is `_pin_sqlite_source:2525` — reached only after `if not private_paths._WINDOWS_PLATFORM: … return` at `:2514-2521`. Verified empirically on this macOS box: wrapping `private_sqlite._prepare_artifact` with a counter and running `connect_private_sqlite("db.prompts.primary", <tmp>/x.db)` → `in-process _prepare_artifact calls during a POSIX open: 0`, `_WINDOWS_PLATFORM: False`, `_posix_guards_available: True`. POSIX opens go `_connect_registered_sqlite` → `prepare_in_helper` (`:1878`) → helper process → `private_sqlite_files.prepare_batch` (`private_sqlite_helper.py:272`).
- Why it matters: this is the code that decides whether a SQLite file is a safe private artifact (O_NOFOLLOW|O_EXCL open, `st_nlink == 1`, uid, 0600, re-stat postcondition). Every macOS/Linux user runs only the `private_sqlite_files` copy; the 440 lines in `private_sqlite.py` are exercised on Windows alone and by nothing in the POSIX test runs. A hardening fix applied to the copy a reader happens to open leaves the other platform unprotected, and the five already-divergent bodies are the mechanism by which that happens.
- Recommended correction: delete the `private_sqlite.py` copies and import the `private_sqlite_files` ones, passing `_pin_job`'s preflight callables through the injection seams that module already has (`open_artifact_fd`/`postcondition_holds`/`identity_out`) — that is exactly the "staged migration" its docstring describes, and it makes the `:22` import live. If the seams cannot express `preflight_body_errors`, add one more parameter rather than a second copy.
- Size: M · ADR: no (no `backlog/decisions/` entry covers this split; `private_sqlite_files.py`'s own docstring already states the intended end state) · Confidence: verified
- Pinning test: `Tests/DB/test_private_sqlite.py` patches `private_sqlite_files._open_artifact_fd` in 14 places to drive the TOCTOU races — i.e. the race tests exercise the **helper** copy only; nothing patches `private_sqlite._open_artifact_fd`. `Tests/Packaging/test_private_sqlite_helper_distribution.py:28` pins `private_sqlite_files.py` as a shipped helper file, so the module must stay — only the duplicate must go.
- Already covered: none

### P3 [D4b] — `EvalsDB` writes two incompatible textual timestamp shapes into the same `updated_at` columns; no reader compares them today, so this is the pattern one query away from the sibling P1s, not a live defect
- Where: `tldw_chatbook/DB/Evals_DB.py` — SQL side, 28 sites `DEFAULT (datetime('now', 'utc'))` / `SET updated_at = datetime('now','utc')` (e.g. `:281-282`, `:339-340`, `:523-524`, `:902`, `:983-984`, `:1331`, `:1363-1364`); Python side, 7 sites `datetime.now(timezone.utc).isoformat()` (`:1619`, `:1666`, `:1850`, `:2312`, `:2364-2367`). `eval_runs.updated_at` gets the SQL shape on INSERT (`:339-340` default) and the Python shape on every status change (`:1619`) and every stored result (`:1850`); `ab_tests.updated_at`/`completed_at` likewise (`:2312`, `:2364-2367`).
- Evidence: `TZ=America/Los_Angeles $PY -c` on SQLite 3.49.1 →
  `datetime(now,'utc')  : 2026-09-18 14:51:24` · `python isoformat     : 2026-09-18T14:51:24.928673+00:00` · `lexical a<b ? True`.
  Space (0x20) sorts before `T` (0x54), so a plain string compare puts **every** SQL-written row before **every** Python-written row regardless of real time; and `datetime.fromisoformat` returns a *naive* datetime for the first and an *aware* one for the second (subtracting one from the other raises `TypeError`).
- Why it is only P3 today (traced, not assumed): every `ORDER BY` in the file is on `created_at` (`:1118`, `:1162`, `:1271`, `:1511`, `:1777`, `:1999`, `:2440`), which is written **only** by the SQL default — single shape. `deleted_at` is used solely as `IS NULL` / `IS NOT NULL`. `updated_at` is never ordered, compared, or parsed. The one Python parse of an Evals timestamp, `Evals/eval_orchestrator.py:821-822` `datetime.fromisoformat(run_data["start_time"]) / ["end_time"]`, reads two columns that are **both** Python-written (`Evals_DB.py:1619-1640`), so it is currently safe — but `end_time` is in `update_run`'s `allowed_fields` (`:1667-1675`), i.e. a caller-supplied value reaches it unnormalized.
- Contrast that proves the repo already knows this class: `Subscriptions_DB` has exactly the same mixture (43 × `CURRENT_TIMESTAMP` + 6 × Python `isoformat()`) and handles it — every reader pushes **both sides through SQLite `datetime()`** and the reason is written down at `Subscriptions_DB.py:1322-1324` and `:4427-4432` ("a bare string compare orders ' ' before 'T' (PR #1443 review)"). `AgentRuns_DB` avoids it differently: one writer, `_now_iso()` at `:145-146` (`%Y-%m-%dT%H:%M:%S.%fZ`), zero SQL-side timestamp functions — `rg -o "CURRENT_TIMESTAMP|datetime\('now'" AgentRuns_DB.py` → no hits, so its 5 `created_at`/`updated_at` comparisons (`:1096`, `:1256`, `:3164`, `:3197`, `:3038`) are single-format and correct.
- Recommended correction: pick one shape for `EvalsDB` — cheapest is to route the 7 Python sites through a module-level `_now()` that emits SQLite's `'%Y-%m-%d %H:%M:%S'` shape (matching the 28 defaults) rather than converting 28 DDL defaults; until then, any new predicate on `updated_at`/`completed_at` must wrap both sides in `datetime()` as Subscriptions does.
- Size: S · ADR: no · Confidence: verified (formats and lexical order measured; the "no reader compares them" half is a traced read of every `ORDER BY`/comparison in the file)
- Pinning test: none
- Already covered: none

### P2 [D4a] — `EvalsDB._loads_json_or_default` was added to stop a NULL JSON column raising `TypeError` out of a lookup, then applied to only 4 of the 15 JSON-column reads in the same file; the 11 raw ones include three nullable columns
- Where: helper at `tldw_chatbook/DB/Evals_DB.py:1457-1476`. Adopted at `:1492` (`get_model`), `:1518` (`list_models`), `:1720` (`get_run`), `:1784` (`list_runs`). **Not** adopted at `:1101` `get_task`, `:1125` `list_tasks`, `:1188` `search_tasks`, `:1260` `get_dataset`, `:1279` `list_datasets`, `:1410` `search_datasets`, `:2007-2010` `get_run_results` (four columns), `:2141` `list_probe_turn_annotations`, `:2395/:2397` `get_ab_test`, `:2447/:2449` `list_ab_tests`.
- Evidence (isolated env, temp-file DB; one row inserted with `metadata` NULL, which the schema at `:298` permits — `metadata TEXT` with no NOT NULL):
  `get_dataset RAISED: TypeError the JSON object must be str, bytes or bytearray, not NoneType` / `list_datasets RAISED: TypeError …` / `search_datasets RAISED: TypeError …`, while `EvalsDB._loads_json_or_default(None, {}, column="x") -> {}`. `eval_results.logprobs/metrics/metadata` (`:355-360`) are nullable too and read raw at `:2007-2010`.
- Why it matters: the helper's own docstring says why it exists — *"Rows created without their config columns carry NULL, and `json.loads(None)` raises TypeError out of lookup APIs whose consumers include the Evals screen"* (TASK-21519). That is still true for datasets and results; only models and runs were fixed. Today's in-module writers always pass `json.dumps(x or {})`, so the NULL has to arrive from a restored/recovered database (`Evals/recovery.py` recreates these tables) or an older row — which is exactly the case TASK-21519 was filed for. The second half of the helper (raising `EvalsDBError` naming the column instead of a bare `ValueError` for corrupt JSON, PR #2634) is also missing at those 11 sites.
- Recommended correction: route the 11 remaining reads through `_loads_json_or_default` with the right default (`{}` for the config/metadata columns, `[]` for `eval_probe_turn_annotations.tags`). Mechanical.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none asserting the raw form.
- Already covered: none

### P3 [D1] — `EvalsDB.search_tasks`'s LIKE branch does not escape `%`/`_`, and that branch is selected precisely *because* the query contains punctuation
- Where: `tldw_chatbook/DB/Evals_DB.py:1150-1166` — the branch guard is `if len(query) <= 2 or any(not (c.isalnum() or c.isspace()) for c in query):` and the query is then bound as `(f"%{query}%", f"%{query}%", limit)` with no escaping and no `ESCAPE` clause.
- Evidence (isolated env, temp-file DB with two tasks `alpha` and `beta`): `search_tasks('_') -> ['alpha', 'beta']`, `search_tasks('%') -> ['alpha', 'beta']`, `search_tasks('zz') -> []`. The control-char filter at `:1148` keeps `%` and `_` (both printable), and the guard at `:1153` routes any string containing them to the LIKE branch.
- Why it matters: a user typing a literal `_` or `%` into the bench search box gets every task back instead of the ones containing that character. Not an injection (the value is bound), just wrong results. `Evals_DB` is the only DB module in this slice with **no** LIKE-escape helper at all — see the P2 [D4b] LIKE finding above; adding the shared `escape_like` would fix this site as a side effect.
- Recommended correction: escape the value and add `ESCAPE '\'`, as `Prompts_DB.browse_prompts` (`:3258-3264`) already does.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P2 [D3] — `AgentRunsDB._METADATA_COLUMNS` says it is "Every `agent_runs` column EXCEPT `steps`" and is not: the v21 routing-snapshot columns were added to the table and to `SELECT *` but not to this list, so `get_run_metadata()` silently returns a smaller dict than `get_run()`
- Where: `tldw_chatbook/DB/AgentRuns_DB.py:1671-1682` (the constant + its comment "Every ``agent_runs`` column EXCEPT ``steps``"); the v21 columns added at `:483-486` (DDL) and `:829-835` (guarded ALTER); readers at `:2600` (`get_run_metadata`) and `:2759` (`list_recent_metadata`-style page). `get_run` at `:2563` uses `SELECT *` and does carry them.
- Evidence (isolated env, temp-file DB, one run created with all four routing fields set):
  `agent_runs columns: 23` · `_METADATA_COLUMNS : 18` · `missing from _METADATA_COLUMNS: ['resolved_base_url', 'resolved_model', 'resolved_params_json', 'resolved_provider', 'steps']`
  `get_run keys` includes all four `resolved_*`; `get_run_metadata keys` does not.
- Why it matters: `get_run_metadata`'s own docstring (`:2568-2589`) instructs callers to "Use this instead of :meth:`get_run` wherever the caller only inspects status/budget/result/task/etc", and two call sites were already migrated on that advice. The next caller that also wants the pinned route gets `KeyError` — or, if it uses `.get()`, a silent `None` that falls back to **live re-resolution of a possibly-edited preset**, which is precisely the failure ADR-147/TASK-32477 added these columns to prevent. Nothing hits it today only because the routing snapshot has its own dedicated reader (`:2694-2707`).
- Recommended correction: either add the four columns to `_METADATA_COLUMNS` (they are small TEXT values; the constant exists to avoid fetching the big `steps` blob, not these) or change the comment to name every excluded column. Then add the drift guard: `rg -n "_METADATA_COLUMNS" Tests/` → no hits, so nothing fails when the next `ALTER TABLE agent_runs` lands. The file already has the precedent — `Tests/DB/test_agent_runs_db.py::test_schema_version_constant_agrees_with_the_version_table` exists for exactly this class of constant-vs-schema drift (class docstring, `:245-256`).
- Size: S · ADR: no (ADR-147 defines the columns, not this constant) · Confidence: verified
- Pinning test: none (`rg -n "_METADATA_COLUMNS" Tests/` → no hits)
- Already covered: none

### P1 [D2] — `AgentRunsDB.reconcile_orphaned_runs` re-scans the ENTIRE terminal run history on the first construction per process, issuing one `agent_run_steps` query per run and JSON-parsing every step payload, inside a `BEGIN IMMEDIATE` write transaction — and one of the construction sites is a Textual `compose()`
- Where: `tldw_chatbook/DB/AgentRuns_DB.py:2340-2519`. The cost is the second half: `:2494-2497` selects **every** non-running primary/subagent row (`SELECT id, status FROM agent_runs WHERE status != 'running' AND agent_kind IN ('primary','subagent')` — no time bound, no LIMIT), then `:2508` calls the closure `run_observations(row["id"])` (`:2391-2419`) per row, which runs `SELECT payload FROM agent_run_steps WHERE run_id = ?` (`:2394-2397`) and `json.loads` on every payload. Called unconditionally from `__init__` (`:283-287`).
- Evidence (isolated env, temp-file DB, steps carrying the expected terminal `kind` so the diagnostic INSERT does *not* fire — i.e. this is the steady-state cost, not a one-off repair):
  - 5000 runs × 50 steps (250 000 step rows, 119 MB file): `first open : open+reconcile = 326.2 ms` · `second open : open+reconcile = 331.9 ms` · `guarded : open (sweep skipped) = 34.2 ms` → **~295 ms of reconcile on every process's first open**, repeated identically on the next launch.
  - Linear in history: 200 runs×20 = 46.5 ms, 1000×20 = 83.3 ms, 2000×20 = 136.8 ms (same script, `_swept_paths` cleared between runs).
- Why it matters: `AgentsSettingsPanel.__init__` calls `_derive_runs_db(app_instance)` → `AgentRunsDB(...)` (`Widgets/settings_agents_panel.py:131-146`, `:177`), and that panel is constructed inside `settings_screen.py:20834` `yield AgentsSettingsPanel(self.app_instance, id="settings-agents-panel")` — a `compose()` body, i.e. the Textual event loop. Whichever construction happens first in the process pays the full sweep; when that is Settings ▸ Agents, the UI is frozen for the duration and the `BEGIN IMMEDIATE` also holds the single SQLite write lock against any concurrent agent writer. (`Chat/console_runtime.py:3458` is the well-behaved site — its docstring says "Call via `asyncio.to_thread`".)
- Recommended correction: two independent fixes, both small. (1) Replace the per-run `run_observations` query with the class's own `_batch_hydrate_steps` (`:1687-1723`) — it exists for exactly this and already chunks at `_IN_CLAUSE_CHUNK`; the file's own docstring calls no-N+1 "this file's existing no-N+1 precedent (e.g. TASK-1972's conversation-level `change_snapshots` fetch)". (2) Bound the terminal-row scan — the lifecycle-capture backfill only needs rows that could plausibly predate the capture code, so an `AND created_at > ?` watermark (persisted like the `schema_version` audit rows) turns a full-history scan into an incremental one. Failing both, move the sweep off the event loop at the `compose()` site.
- Size: M · ADR: no · Confidence: verified (timings measured; the `compose()` reachability is a traced read of the two files named)
- Pinning test: none measuring cost. `rg -n "reconcile_orphaned_runs" Tests/` — see dispositions.
- Already covered: none

### P3 [D1] — the "Chats with unavailable characters" filter lowercases one side in Python (`str.casefold`, full Unicode) and the other in SQLite (`LOWER()`, ASCII-only), so a non-ASCII search term matches nothing
- Where: `tldw_chatbook/DB/character_conversation_search.py:1378-1379` and `:1526-1527` — `query_pattern = f"%{escaped_query.casefold()}%"`, compared against `LOWER(COALESCE(c.title, '')) LIKE ? ESCAPE '\'` (and three sibling columns) at `:1390-1394` / `:1538-1542`.
- Evidence: `$PY -c` on SQLite 3.49.1 —
  `sqlite LOWER(Ärger) = 'Ärger'` · `python casefold = 'ärger'` · `match? 0` · (ASCII control `Alpha` → `1`) · `'Straße'.casefold() = 'strasse'` → `0`.
  SQLite's built-in `LOWER()` only folds A-Z unless the ICU extension is loaded; `str.casefold()` folds the full Unicode range **and** expands `ß`→`ss`.
- Why it matters: typing the exact title of a chat named "Ärger" (or any term with a non-ASCII letter, or `ß`) into the unavailable-characters filter returns zero rows, on both the count (`_unavailable_total`) and the page (`_unavailable_sources`) — the user sees "0 results" for a chat that is right there. The same asymmetry makes the count and page consistent with each other but both wrong.
- Recommended correction: fold one side only. Cheapest is to drop `.casefold()` and rely on SQLite's `LIKE`, which is already case-insensitive for ASCII by default — that makes both sides use the identical (ASCII) rule and removes the mismatch. A genuinely Unicode-correct filter needs a `create_function("cc_lower", 1, str.casefold, deterministic=True)` on both sides, the pattern `Prompts_DB.browse_prompts:3299-3301` already uses for its own sort/filter.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P2 [D4b] — the copy-pasted held-connection store template drifted: 7 stores' `transaction()` catches `Exception`, so a `BaseException` inside the block skips the rollback and WEDGES that thread's connection for the rest of the process; 4 stores already catch `BaseException` and one of them documents exactly why
- Where (the template, `_held_connection` / `close` / `transaction`, copied verbatim): `DB/Workspace_DB.py:392/:464/:434`, `DB/AgentRuns_DB.py:341/:387/:403`, `DB/RAG_Indexing_DB.py:155/:242/:214`, `DB/Library_Collections_DB.py:550/:683/:651`.
  Handler sweep over every `transaction()`/`read_transaction()` that issues a BEGIN (`ast` scan of `tldw_chatbook/DB/*.py`):
  `except Exception` → `AgentRuns_DB.py:403`, `Library_Ingest_Jobs_DB.py:65`, `Prompts_DB.py:659`, `RAG_Indexing_DB.py:214`, `Subscriptions_DB.py:1987`, `Workspace_DB.py:434`, `Client_Media_DB_v2.py:1325`.
  `except BaseException` → `Library_Collections_DB.py:601` (`read_transaction`) and `:651` (`transaction`), `Workflows_DB.py:50`, `automatic_work.py:60`.
- Evidence (isolated env, temp-file DBs, `KeyboardInterrupt` raised inside the `with` body):
  `RAG: conn.in_transaction after interruption = True` → `RAG: next transaction() RAISED OperationalError: cannot start a transaction within a transaction`
  `LibCollections: conn.in_transaction after interruption = False` → `LibCollections: next transaction() OK`
  Also verified by AST diff that the three `_held_connection` bodies and three `close` bodies are byte-identical across Workspace/AgentRuns/RAG_Indexing, and that the only differences in `transaction()` are the dropped `immediate` parameter and this handler.
- Why it matters: these are *held* connections in autocommit mode (`isolation_level = None`) with an explicit `BEGIN IMMEDIATE`. Missing the rollback leaves the transaction open on the thread-local connection, so (a) the uncommitted writes are silently discarded at close, and (b) **every subsequent `transaction()` on that thread raises** until the process ends — the store is bricked, not merely degraded. `Library_Collections_DB.transaction`'s docstring already names the case: *"Re-raised after rolling back, on any error **or interruption** inside the `with` block."* One store's author saw this; the fix was never propagated back to the template's other copies.
- Reachability, stated honestly: I could not find a shipped caller that raises a `BaseException` inside one of these blocks — an AST sweep of every `with <x>.transaction()` block in `tldw_chatbook/` found **0** containing an `await` (so `asyncio.CancelledError`, a `BaseException` since 3.8, cannot land mid-block today). The live triggers are therefore `KeyboardInterrupt` and `SystemExit` at interpreter shutdown, and any future `await`/cancellation inside such a block. That is what keeps this P2 rather than P1 — the *mechanism* is verified, the *shipped trigger* is not.
- Wider than `transaction()`: an AST sweep for every `try` block that contains a `ROLLBACK`/`rollback` next to a `BEGIN IMMEDIATE` and whose handlers are `Exception`-only found **18 sites** across 8 modules — `AgentRuns_DB.py:423`, `Chunking_Lab_DB.py:98/:414/:525/:590/:628` (this store issues `BEGIN IMMEDIATE`/`COMMIT`/`ROLLBACK` by hand, same shape), `Prompts_DB.py:673`, `RAG_Indexing_DB.py:234`, `Subscriptions_DB.py:1196/:1422/:2051`, `Workspace_DB.py:456/:731/:741/:751/:761`, plus `ChaChaNotes_DB.py:24028` and `Client_Media_DB_v2.py:1349` outside this slice.
- Recommended correction: change the `except Exception:` guards to `except BaseException:` — the same one-word edit `Library_Collections_DB`, `Workflows_DB` and `automatic_work` already carry. The deeper fix is that this template is copy-pasted at all: `DB/base_db.py` is the shared base every one of these classes already subclasses, and `_held_connection`/`close`/`transaction` belong there once (the three-store byte-identical bodies prove they can be).
- Size: S for the handler fix, M to hoist the template into `base_db.py` · ADR: no · Confidence: verified
- Pinning test: none — no test raises a `BaseException` inside a `transaction()` block.
- Already covered: none

### P3 [D1] — `RAGIndexingDB.needs_reindexing` raises `TypeError` on a naive `datetime`, the same shape its own writer silently accepts
- Where: `tldw_chatbook/DB/RAG_Indexing_DB.py:827-828` — `last_modified = datetime.fromisoformat(info["last_modified"]); return current_modified > last_modified`. The stored value is always tz-aware (the process-wide adapter in `DB/sqlite_datetime_fix.py:12-20` stamps UTC onto a naive input), but `current_modified` is whatever the caller passes.
- Evidence (isolated env, temp-file DB): `mark_item_indexed("i1","media", datetime(2026,9,18,12,0,0), 3)` → `stored last_modified: '2026-09-18T12:00:00+00:00'`; `needs_reindexing("i1","media", <same naive value>)` → `RAISED TypeError can't compare offset-naive and offset-aware datetimes`; with the aware value → `False`.
- Why it is only P3: the one shipped caller, `RAG_Search/ingestion_indexing.py:745`, feeds it `entry.last_modified`, which `_coerce_timestamp` (`:533-556`) has already forced tz-aware, **and** wraps the call in `except Exception` (`:748-752`) that logs and indexes anyway. So today the worst case is a silent loss of the incremental skip, not a crash. The asymmetry is still a live trap: the write side accepts naive input without complaint and the read side rejects it.
- Recommended correction: normalize inside `needs_reindexing` — `if current_modified.tzinfo is None: current_modified = current_modified.replace(tzinfo=timezone.utc)`, matching what the storage adapter already does on the write path. One line.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none


## Candidate dispositions

| candidate (file:line pattern) | disposition |
|---|---|
| dup_shape `transaction@Library_Collections_DB:651 / Library_Ingest_Jobs_DB:65 / AgentRuns_DB:403 / RAG_Indexing_DB:214` | **confirmed** → P2 [D4b] store-template drift (`except Exception` vs `except BaseException`), verified repro |
| dup_verbatim `_held_connection@Workspace_DB:392 / AgentRuns_DB:341 / RAG_Indexing_DB:155` | **confirmed** — AST-diff says byte-identical; folded into the same P2 |
| dup_verbatim `close@Workspace_DB:464 / AgentRuns_DB:387 / RAG_Indexing_DB:242` | **confirmed** byte-identical; `Library_Collections_DB:683` differs only in `pass` vs `return` as the last statement — **no behavioural drift** (retired as a defect, kept as evidence of the copied template) |
| dup_verbatim `transaction@AgentRuns_DB:403 / RAG_Indexing_DB:214` | confirmed (see above) |
| dup_shape `add_keyword@Prompts_DB:1271 / get_conversation_active_leaf@ChaChaNotes:12707 / _provider_config@settings_screen:12619 / release@pending_handoff_store:445` | **retired** — a shape-only match (try/transaction/except); the four bodies share no logic. Read all four headers; nothing to consolidate |
| dup_shape `_library_prompt_fts_query@Prompts_DB:3330` + 3 siblings | **confirmed but out of scope here** — 4 copies of the same tokenize+`quote_fts5_token` builder across Prompts/Media/ChaChaNotes. The escape itself already lives in the shared `Utils/fts5_match_forms`; only the 6-line wrapper is re-rolled. P3, subsumed by the LIKE-escape D4b's recommendation (same canonical home) |
| dup_shape `_library_keywords_for_prompts@Prompts_DB:3343` + 3 siblings | **confirmed, P3** — four per-store keyword fan-in helpers with different table names; genuinely per-store SQL, only the grouping loop repeats |
| dup_shape/verbatim `__str__@Prompts_DB:107 / Client_Media_DB_v2:114 / ChaChaNotes:409` | **confirmed, P3** — three identical `ConflictError.__str__`. Related live defect filed separately (the `soft_delete_keyword` mis-call) |
| dup_verbatim `get_schema_version@Library_Collections_DB:757 / Workspace_DB:810` | **confirmed, P3** — two identical `SELECT MAX(version) FROM schema_version` wrappers; belongs in `base_db.py` with the rest of the template |
| dup_verbatim `relocate@recovery_operations:106 / recovery_core:276` | **confirmed, P3 (trivial)** — AST diff: bodies identical (2 statements), only one carries the docstring |
| dup_verbatim `_artifact_postcondition_holds / _path_error_from_oserror / _optional_sidecar_restart_or_absent @private_sqlite vs private_sqlite_files` | **confirmed** → P2 [D4b] duplicated artifact validator (9 shared functions, 5 already drifted, dead import at `:22`) |
| except_exception_pass `Library_Collections_DB:693` | **retired** — documented best-effort teardown (`# noqa: BLE001 - best-effort teardown`), last statement of `close()`, identical effect to the siblings' `return` |
| except_exception_pass `private_sqlite:1592 / :2310 / :2643` | **retired** — all three are cleanup paths with an explicit comment saying the primary failure is retained elsewhere (`__del__` retirement; "Original native failure is retained by the observer"; a `warnings.warn` that itself must not raise) |
| except_exception_pass `Subscriptions_DB:235 / :239 / :285` | not examined (file completed by the earlier run) |
| except_exception_return `AgentRuns_DB / RAG_Indexing_DB / Workspace_DB` (1 each) | **retired** — all three are the same `close()` line, commented "preserve retryable native cache"/"preserve explicit retirement route": a failed `conn.close()` deliberately keeps the thread-local reference so the connection is not lost while still live |
| except_exception_return `character_conversation_search` (2) | **retired** — `_apply_dirty_generation:1227` ("maintenance is a typed boundary") and `_reconcile_generation:1302` ("reconciliation is a typed boundary"), plus `ensure_keyword_index:1047` ("persist a typed failed generation for callers") — each returns a `CharacterKeywordIndexStatus`; the status enum is the contract |
| except_exception_return `sql_logging` (1) | **retired** — `preview_params` returns `"<unrepr-able params>"`; documented ("Debug-log preview building must never break the query path") |
| fetchall_dynamic_sql `Prompts_DB:3943 search_prompts` | **confirmed** → P1 (unbounded id set + parameter cliff) |
| fetchall_dynamic_sql `Prompts_DB:4539 search_prompts_by_text` | **confirmed, same defect** — `:4523-4539` repeats the `IN ({placeholders})` shape over an unbounded FTS rowid set, and additionally returns **every** match with full rows plus an N+1 `fetch_keywords_for_prompt` per row. Folded into the P1's recommendation |
| fetchall_dynamic_sql `Prompts_DB:4439 search_prompts_by_keyword` | **confirmed, P2** — no LIMIT and an N+1 keyword fetch per row (`:4458-4461`); same fix family |
| fetchall_dynamic_sql `Prompts_DB:3122 list_prompts`, `:3359`, `:3754`, `:3772`, `:4099`, `:4312`, `:4369`, `:4398`, `:4829`, `:4877`, `:4955` | **verified-fine (bounded)** — `list_prompts`/`get_all_*` take `LIMIT ? OFFSET ?`; `_library_keywords_for_prompts` is bounded by a page's id list; history/sync-log take explicit limits; the three export helpers are whole-library-by-design (the user asked to export everything) |
| fetchall_dynamic_sql `AgentRuns_DB:2811 list_runs` | **verified-fine** — `limit` is optional but the docstring and `count_runs` (`:2853-2889`) exist precisely so callers page; the only unbounded caller is `subagent_runs`, whose size is a conversation's own run count |
| fetchall_dynamic_sql `AgentRuns_DB:1101`, `:2030`, `:2851`, `:2913` | **verified-fine** — `list_unseen_console_activity` is keyset-paged with a hard 500 cap; `list_agent_definitions` is bounded by user-authored definitions; `list_subagent_run_headers` validates `1..101`; `list_running_run_ids` is bounded by concurrently-running runs |
| fetchall_dynamic_sql `Evals_DB:1123 / :1516 / :1782 / :2445` | **verified-fine** — all four take `limit`/`offset` and append `LIMIT ? OFFSET ?` |
| fetchall_dynamic_sql `agent_worktrees:211 list_for_conversation` | **verified-fine** — `ORDER BY worktree.run_id ASC LIMIT ?` (`:215`) |
| fetchall_no_limit `AgentRuns_DB:2452 / :2460 / :2505 reconcile_orphaned_runs` | **confirmed** → P1 (full-history scan + per-run step query on the first construction per process) |
| fetchall_no_limit `AgentRuns_DB:674/:729/:741/:772/:817` (PRAGMA table_info) | **retired** — `PRAGMA table_info` returns one row per column |
| fetchall_no_limit `AgentRuns_DB:1268/:1308/:1332/:1355/:1501/:1523/:1557/:1591/:2310/:2396/:2926/:3155/:3177/:3191/:3280` | **verified-fine** — each is scoped by `run_id` or `conversation_id`; `:3155/:3177/:3191` join parent↔child and are bounded by a conversation's sub-agents; `:3280` is the batched `GROUP BY` that replaced an N+1 (documented at `:3253-3266`) |
| fetchall_no_limit `Evals_DB:973/:1979/:2055/:2139/:2200/:2224` | **verified-fine** — `:973` distinct run groups of one task; `:1979` one aggregate row per run group (documented as deliberately O(1) round trips); the rest are keyed by `run_id`/`run_group_id` or by a caller-supplied id list |
| fetchall_no_limit `Prompts_DB:755/:816` (PRAGMA), `:1177/:1792/:2182/:2830/:2884/:2924/:3300/:3453/:3546/:3884/:3901/:3912/:4227/:4523` | `:3884/:3901/:3912/:4523` **confirmed** (part of the P1); the rest **verified-fine** — per-prompt keyword fan-ins, COUNT(*) queries, and `get_all_active_prompt_ids` (documented "uncapped ID query", ids only) |
| fetchall_no_limit `Library_Ingest_Jobs_DB:477 all_jobs` | **verified-fine** — one caller (`app.py:3060`), bounded by the user's own queued ingest jobs |
| fetchall_no_limit `Chunking_Lab_DB:608/:615`, `Workspace_DB:608/:616/:643`, `private_sqlite:3694`, `recovery_core:270`, `recovery_sqlite:54`, `VisualIdentity_DB:47/:165` | **verified-fine** — PRAGMAs, single-row lookups, or per-pack/per-version reads |
| fetchall_no_limit `character_conversation_search:102 _project` | **verified-fine** — loading a conversation's whole message set is inherent to walking its branch graph |
| fetchall_no_limit `character_conversation_search:409/:702/:1479` | **verified-fine** — `COUNT(*)` queries |
| fetchall_no_limit `Subscriptions_DB:*` (16 rows) | not examined (file completed by the earlier run) |
| function_body_import_per_file `recovery_operations (23)`, `recovery_core (14)`, `private_sqlite (15)`, `recovery_sqlite (2)` | **retired** — deliberate and stated: `recovery_operations.py`'s module docstring opens *"Installed operational SQLite policies; imports never open runtime stores."* Hoisting these would make importing a policy catalog import the live connection seam |
| function_body_import_per_file `Prompts_DB (10)` | **partly confirmed, P3** — 8 are `import csv/tempfile/os/zipfile/datetime` inside the three export helpers (stdlib, cheap, arguably local-by-choice) but `_add_keyword_with_retry:4607` and `execute_query_with_retry:4641` each do `import time` *inside the retry loop's except branch* while the module already has `import time` at `:34` — dead per-iteration work and a shadowing hazard |
| function_body_import_per_file `AgentRuns_DB (1)`, `Library_Collections_DB (2)`, `Library_Ingest_Jobs_DB (1)`, `automatic_work (1)`, `sqlite_datetime_fix (1)` | **retired** — cycle breaks (`automatic_work` ↔ `AgentRuns_DB` is `TYPE_CHECKING`-guarded at module level and imported lazily in `cached_property automatic_work`) or one-time setup |
| lock_and_execute `Subscriptions_DB:0 locks=2 executes=211` | not examined (file completed by the earlier run) |
| mutable_class_attr `AgentRuns_DB:266 _swept_paths` | **retired** — deliberate and commented (`# DB files already reconciled this process`); process-wide by design, and a check-then-add race only costs one redundant idempotent sweep |
| raw_1024x1024 `Chunking_Lab_DB:38/:39`, `private_sqlite:1667` | **retired** — named constants (`_MAX_SAMPLE_BYTES`, `_MAX_RESULT_BYTES`) and a read chunk size |
| seed_name `_get_connection` ×7, `_held_connection` ×4, `_initialize_schema` ×7, `_identity` ×2, `_now`, `_now_iso` | **confirmed as a cluster** → the store-template P2; individually unremarkable |
| strftime `AgentRuns_DB:146` | **verified-fine** — single writer, no SQL-side timestamp anywhere in that file, so its 5 `created_at`/`updated_at` comparisons are single-format |
| strftime `Prompts_DB:1016` | **verified-fine** — `_get_current_utc_timestamp_str` is the file's only timestamp writer; no SQL-side `CURRENT_TIMESTAMP`/`datetime('now')` in `Prompts_DB.py` |
| strftime `Prompts_DB:4807 / :4968` (`%Y%m%d_%H%M%S`) | **confirmed** → the predictable temp-filename P2 |
| tempfile_no_secure `Prompts_DB:4808 / :4973 / :5007 / :5009` | **confirmed** (3 of 4) → the predictable temp-file P2; `:5007` `tempfile.mkdtemp()` is correct and retired |
| try_import_guard `private_sqlite:2999 / :3120` (`BaseException`) | **retired** — not optional-dependency guards; they are the cleanup-owner blocks of `discard_profile_migration_destination` / `open_canonical_profile_migration_destination`, where catching `BaseException` is the point |
| legacy_markers_per_file (9 files) | **retired as a class** — sampled `AgentRuns_DB` (15) and `Prompts_DB` (9): every marker is a dated schema-migration comment (`v10->v11`, "legacy blob column") documenting a still-required compatibility path, not dead code |

## Verified-fine
- **Evals_DB's task-22224 exception is honest.** Its `_get_connection` docstring (`:196-215`) claims every write path uses `with conn:`; the AST DML scan confirms it (the only "bare" non-DDL DML is the FTS rebuild in `_migrate_schema`, reached solely from `_init_schema`'s `with conn:`). No lost-write shape.
- **Six stores are simply not on legacy isolation**, so the P0 shape cannot occur: `Workspace_DB:383`, `Library_Collections_DB:529`, `AgentRuns_DB:332`, `RAG_Indexing_DB:144`, `Library_Ingest_Jobs_DB:102`, `Chunking_Lab_DB:96`, `Workflows_DB:27` all set `isolation_level = None`.
- **`Subscriptions_DB`'s mixed timestamp shapes are handled, not missed.** Every reader normalizes both sides through SQLite `datetime()`, and the reason is written down at `:1322-1324` and `:4427-4432` ("a bare string compare orders ' ' before 'T' (PR #1443 review)"). Contrast with the Evals P3 above.
- **`sqlite_datetime_fix` is registered process-wide, not per store.** `DB/__init__.py:2` imports it and the module registers adapters/converters at import (`:79`). The apparent "drift" is that converters only fire for connections opened with `detect_types`, and exactly three do: `ChaChaNotes_DB:3482`, `Client_Media_DB_v2:1130`, `Prompts_DB:460` (`rg -n "detect_types" --type py tldw_chatbook`). Each store is internally consistent, so this is a documented per-store choice rather than a defect — but it does mean a `DATETIME` column reads back as `datetime` in three stores and as `str` everywhere else.
- **`datetime('now','utc')` is not a double-conversion bug on this build.** Suspected, then measured: `TZ=America/Los_Angeles` → `datetime('now')`, `datetime('now','utc')` and `CURRENT_TIMESTAMP` all returned the identical UTC string on SQLite 3.49.1.
- **Derived-artifact rules are in sync for the newest migrations.** `chachanotes_v72_to_v73_note_links.sql` adds `note_links` (present in `sql_validation.py:149`) and `idx_note_links_target` (1 row in `scripts/index_plan_pin_census.tsv`); `chachanotes_v70_to_v71` adds `idx_conversations_archive` (1 census row); `chachanotes_v69_to_v70`'s `buddy_profiles`/`buddy_visual_bindings` are at `sql_validation.py:72-73`.
- **`preview_params` (`sql_logging.py`) does the right thing** — bytes summarized by length, per-item and whole-preview caps, and an explicit note about why an "iterable" branch is absent (it would consume a generator before the query ran).
- **`Prompts_DB._apply_migration_v3_to_v4`'s index-predicate check is not a case bug.** `_PROMPT_HISTORY_INDEX_PREDICATE` is lowercase while the DDL is mixed case, but the comparison lowercases `sqlite_master.sql` first (`:918`).

## Retired
- **"`_build_existing_writable_uri` could corrupt a path containing `?mode=ro`"** (`private_sqlite.py:1279-1290`, a blind `.replace(..., 1)`). Retired: the URI is built by `Path(raw).as_uri()`, which percent-encodes `?` as `%3F`, so the appended query string is the only match.
- **"`Library_Collections_DB.close()` drifted from the template and loses the connection reference on a failed close."** Retired: AST diff shows the assignment sits *after* `conn.close()` inside the `try`, so a raising close skips it exactly as the siblings' `return` does. Same behaviour, different spelling.
- **"`AgentRunsDB._swept_paths` is a shared-mutable-class-attribute bug."** Retired with the comment and the code: it is a deliberate process-wide registry and the only race outcome is a redundant idempotent sweep.
- **"`Evals_DB` writes `datetime('now','utc')`, which double-converts and shifts timestamps by the local UTC offset."** Retired by measurement (see Verified-fine) — a no-op on SQLite 3.49.1. The *format* mismatch survived as the P3.
- **"`recovery_operations`' 23 function-body imports are a D3 smell."** Retired against the module docstring's stated contract.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| The `except Exception` transaction guards are reachable from a shipped path (would promote the P2 to P1) | I proved the mechanism with `KeyboardInterrupt` and proved `asyncio.CancelledError` cannot land today (0 `await`s inside any `with …transaction()` block), but I did not audit `SystemExit`/`GeneratorExit` at shutdown | `cd $WT && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY -c "import ast,glob,pathlib; print([ (p,n.lineno) for p in glob.glob('tldw_chatbook/**/*.py',recursive=True) for n in ast.walk(ast.parse(pathlib.Path(p).read_text())) if isinstance(n,(ast.With,ast.AsyncWith)) and '.transaction(' in ' '.join(ast.unparse(i.context_expr) for i in n.items) and any(isinstance(x,ast.Yield) for x in ast.walk(n))])"` (a `yield` inside the block is the `GeneratorExit` route) |
| The `reconcile_orphaned_runs` P1 actually freezes the Settings ▸ Agents UI in the running app | Brief forbids running the app | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify new-session -d …`, open Settings ▸ Agents against a profile whose `agent_runs.db` has ≥5000 terminal runs, `capture-pane` during the transition |
| Whether `scripts/preflight.sh` is currently green for the four derived-artifact checks | Spot-checked the three allowlists by hand instead of running the script (it is a repo-level tool and this review is read-only) | `cd $WT && ./scripts/preflight.sh` |
| `Subscriptions_DB.py` candidate rows (3 `except: pass`, 16 `fetchall_no_limit`, `lock_and_execute`) | The file was read in full by the interrupted earlier run, which filed no finding on them; I did not re-read it | `cd $WT && rg -n "except Exception:\s*$" -A1 tldw_chatbook/DB/Subscriptions_DB.py` |
| Whether the LIKE-escape consolidation would change any current result | The nine copies are byte-identical, so a shared helper is a pure refactor — the TTS `!`-variant is correctly paired (`rg -n "ESCAPE" tldw_chatbook/TTS/profile_repository.py` → `:4678  LIKE ? ESCAPE '!'`), so the two families are each self-consistent and the risk is a future cross-copy | *(settled — no command outstanding)* |
