---
id: TASK-15469
title: 'Personas dictionaries: indexed attachment lookup and a threaded backend'
status: Done
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - personas
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: clicking one dictionary row (`UI/Screens/personas_screen.py:3498-3546` via `_handle_entity_selected:3315`) runs 4+ synchronous queries on the event loop, including `list_dictionary_conversations` (`Character_Chat/local_chat_dictionary_service.py:868-901`): `SELECT id, title, metadata FROM conversations WHERE deleted = 0 AND metadata LIKE '%active_dictionaries%'` — a leading-wildcard LIKE that scans the entire conversations table and JSON-parses matches. The dictionary record is loaded twice per click (`get_dictionary:371` and `get_statistics:760` are each a full load), and `list_dictionaries(include_usage=True)` is N+1 via the same scan per dictionary. With thousands of conversations this is 50-500 ms per click on slow hardware.

Fix direction: thread the local backend in the scope service; replace the LIKE scan with an indexed lookup (JSON1 query with an expression index, or a proper attachment table — a ChaChaNotes schema change means bumping `_CURRENT_SCHEMA_VERSION` with a migration); load the record once per selection and derive statistics from it. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A dictionary row click issues no full-table scan (query-plan or timing evidence) and at most 2 queries, none on the event loop
- [x] #2 Statistics and attachment lists return identical values (tests)
- [x] #3 Click latency before/after with a 1,000+-conversation DB recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
**Mechanism choice (investigated first, as required).** Who writes
`conversations.metadata`? `LocalChatDictionaryService._write_active_dictionaries`
is the only writer of the `active_dictionaries` KEY, but the metadata BLOB is
read-modify-written wholesale by several unrelated paths
(`Chat/chat_persistence_service.py` roleplay context + pinned prefill,
`Chat/rag_scope.py` scope storage, `DB.update_conversation` from anywhere,
sync apply, tests). Option (a) — a JSON1 expression index — cannot work at all
here: `active_dictionaries` is a JSON *array*, and an expression index over
`json_extract(metadata,'$.active_dictionaries')` indexes the array's text, which
answers no membership query. So option (b), an attachment table. But maintaining
it "at attach/detach time" in Python would silently rot the moment any of those
other blob writers changes the key, so it is maintained by SQLite TRIGGERS on
`conversations` (INSERT/UPDATE/DELETE) — every writer, including raw SQL and
sync apply, keeps it correct — plus a backfill in the migration. Same shape as
the existing FTS5 external-content triggers.

**Exact parity, by construction.** SQL cannot reproduce Python's `int()`
coercion for every element shape (probe evidence: `"1_0"`→10, `"٣"`→3,
`1e300`→a 300-digit int, `NaN` literal parses in Python but not in SQLite,
duplicate JSON keys resolve last-wins in Python and first-wins in SQLite). So
the index resolves ONLY elements that are unambiguous JSON integers
(`json_each.type='integer' AND typeof(value)='integer'`); every other shape that
could still coerce in Python marks the conversation in a second, tiny
`conversation_dictionary_unresolved` table whose rows are verified in Python
with the unchanged `_active_dictionaries()` predicate. Element types that can
never coerce (`null`/`object`/`array`) are skipped, matching Python. The trigger
carries today's `metadata LIKE '%active_dictionaries%'` prefilter verbatim so
the new path inherits the old path's blind spots (e.g. a `a`-escaped key)
rather than "fixing" them into a behaviour change.

1. Migration V34→V35: `conversation_dictionary_attachments(conversation_id,
   dictionary_id)` + `idx_..._dictionary`, `conversation_dictionary_unresolved
   (conversation_id)`, three triggers, backfill from existing metadata. Local-only
   (no sync columns, no sync triggers). Bump `_CURRENT_SCHEMA_VERSION` to 35, add
   the runner + `migration_steps` entry, register both tables in
   `DB/sql_validation.py`'s `VALID_TABLES['chachanotes']`.
2. `list_dictionary_conversations`: one indexed query (resolved ∪ unresolved,
   ordered by `conversations.rowid` = today's scan order), unresolved rows
   verified in Python; unresolved verdict always wins over the index.
3. Single load per selection: `statistics_from_record()` (pure) derives the
   `get_statistics` payload from the already-loaded record; `_select_dictionary`
   and the two other record-in-hand call sites use it. `list_versions` accepts the
   preloaded record so `_ensure_history_baseline` stops re-loading it.
4. Threaded backend: `ChatDictionaryScopeService` runs local-mode sync backend
   calls via `asyncio.to_thread` behind a POSITIVE-confirmation predicate (thread
   only when `service.db.is_memory_db is False`); covers select, attachments and
   the entry add/update/delete/reorder handlers, which all route through it.
5. Evidence: parity test over a malformed-metadata fixture corpus, EXPLAIN QUERY
   PLAN assertion (no `SCAN conversations`), a per-click statement counter
   (≤2), an off-loop thread-identity test, and a 1k-conversation latency probe.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The used-by scan is gone, the double load is gone, and the local backend runs off
the event loop. ChaChaNotes schema **V34 -> V35**.

**Schema (V34->V35, `chachanotes_v34_to_v35_conversation_dictionary_attachments.sql`).**
Two derived, local-only tables — `conversation_dictionary_attachments
(conversation_id, dictionary_id)` with an index on `dictionary_id`, and
`conversation_dictionary_unresolved(conversation_id)` — maintained by three
triggers on `conversations` (insert/update/delete) plus a backfill. No sync
columns, no sync_log triggers: this is derived state over `conversations.metadata`,
which is itself synced. The migration header carries the full rationale; the
short version:

* **Triggers, not attach/detach-time maintenance.** Only
  `_write_active_dictionaries` writes the KEY, but the metadata BLOB is
  read-modify-written by `chat_persistence_service`, `rag_scope`, any
  `update_conversation` caller and sync apply. An index maintained in Python at
  attach time would rot the first time one of those wrote it. A trigger cannot be
  bypassed.
* **A JSON1 expression index (option (a)) cannot answer this query at all** —
  `active_dictionaries` is an array, so an index over
  `json_extract(metadata,'$.active_dictionaries')` indexes the array's text, which
  answers no membership question.
* **Two tables, because SQL cannot reproduce Python's `int()` for every element
  shape** (probed, not assumed: `"1_0"`->10, `"٣"`->3, `1e300`->an exact 300-digit
  int, a `NaN` literal parses in Python but is invalid JSON to SQLite, a duplicated
  key is last-wins in Python and first-wins in `json_each`). The index resolves only
  unambiguous JSON integers; every other shape marks the row unresolved and the
  service decides it with the unchanged `_active_dictionaries()` predicate, whose
  verdict overrides the index. Shapes that can never coerce (`null`/`object`/`array`)
  are skipped, matching Python. In an app-written database the unresolved table is
  empty.
* Every `json_*` call in the triggers reads `CASE WHEN json_valid(x) THEN x ELSE
  '{}' END` — `json_each`/`json_type` raise on malformed JSON and a raising trigger
  would fail the conversation write itself. The old query's
  `metadata LIKE '%active_dictionaries%'` prefilter is carried over verbatim so the
  index inherits its blind spots rather than "fixing" them into a behaviour change.
* The pre-existing dead `conversation_dictionaries` junction table is unusable here:
  `conversation_id INTEGER` against a TEXT UUID PK, and an FK to `chat_dictionaries`
  that would fail the write for a stale dictionary id.

**Query.** `list_dictionary_conversations` is one statement over both branches,
ordered by `conversations.rowid` (the old scan's order). Both joins are **CROSS
JOINs**: without them SQLite chose `conversations` as the unresolved branch's outer
loop — i.e. still a full scan (see the lesson added to
`lessons-testing-evidence.md`; it cost 2.07 ms/click at 10k conversations and the
first plan assertion missed it because `EXPLAIN QUERY PLAN` prints the ALIAS).

**One load per selection.** `statistics_from_record()` (new, pure) derives the
`get_statistics` payload from the record already in hand; `_select_dictionary`,
`_refresh_dictionary_statistics` and `_reload_selected_dictionary_entries` use it.
`list_versions(record=...)` seeds a missing history baseline from the same record
instead of re-loading. `get_statistics` itself is unchanged in behaviour (it now
calls the shared helper).

**Threading.** `ChatDictionaryScopeService._call_backend` runs local-mode sync
backend calls via `asyncio.to_thread` behind a POSITIVE-confirmation predicate
(`service.db.is_memory_db is False`) — stricter than task-283's, which threads what
it cannot identify; the doubles behind this seam have no `.db`. This covers
selection, attachments, versions/activity and the entry add/update/delete/reorder
handlers, which all route through the scope service.

**Review bundle (two follow-up fixes, second commit).**
1. The AU trigger cleared only `OLD.id`, but the FK's `ON UPDATE CASCADE` has already
   renamed the index rows to `NEW.id` by the time an AFTER UPDATE trigger fires — so a
   conversation id change left the OLD dictionary ids indexed under the NEW id
   (reproduced: one UPDATE changing id and metadata `[1,2]`→`[3]` left 1, 2 AND 3).
   Both DELETEs now clear `IN (OLD.id, NEW.id)`, which is also what keeps this correct
   on a connection running without `PRAGMA foreign_keys = ON`. The migration file was
   amended in place (it has never merged, so no second migration). Test is
   parametrized over both FK modes and both arms are mutation-checked: `IN (OLD.id)`
   reds only the FK-on arm, `IN (NEW.id)` reds only the FK-off arm.
2. Threading made the version-history sidecar reachable from worker threads, where two
   `_record_history` calls raced on the single `<sidecar>.tmp` write+replace. Closed
   with a `threading.RLock` around every mutate+persist and every read snapshot
   (RLock because `_ensure_history_baseline` nests `_record_history`; no deadlock is
   possible because no caller holds an open DB transaction while taking it). Mutation-
   checked: with the lock stubbed out the new concurrency test fails 3 runs of 3 with
   `FileNotFoundError`.

**Measured** (isolated probe, `add_conversation` + metadata writes, medians of 25):

| conversations | click DB work BEFORE | AFTER | used-by BEFORE | AFTER | used-by, 0 hits, BEFORE | AFTER |
|---|---|---|---|---|---|---|
| 1,500 | 1.11 ms | 0.12 ms | 0.99 ms | 0.07 ms | 0.98 ms | 0.00 ms |
| 10,000 | 7.80 ms | 0.54 ms | 7.74 ms | 0.48 ms | 7.69 ms | 0.00 ms |

BEFORE re-runs the old scan + the old double/triple load against the SAME database.
The old path is O(conversations); the new one is O(matches). Write cost of the
triggers, same probe shape: a metadata-changing `update_conversation` goes 0.097 ->
0.145 ms median; `add_conversation` is unchanged (its metadata has no marker).

**Files.** `DB/migrations/chachanotes_v34_to_v35_conversation_dictionary_attachments.sql`
(new), `DB/ChaChaNotes_DB.py` (version 35 + runner + step), `DB/sql_validation.py`
(both tables allowlisted), `Character_Chat/local_chat_dictionary_service.py`,
`Character_Chat/chat_dictionary_scope_service.py`, `UI/Screens/personas_screen.py`,
`Tests/Character_Chat/test_dictionary_attachment_index.py` (new, 28 tests),
`Tests/UI/test_personas_dictionaries.py` (+1), `Tests/DB/
test_chachanotes_console_context_memory_migration.py` (3 current-version
assertions 34 -> 35), `backlog/docs/lessons-testing-evidence.md`.

**Tests** (after the review bundle). New file 28 passed;
`Tests/Character_Chat/` 573 passed; `Tests/UI/test_personas_dictionaries.py` +
`test_console_dictionaries_screen.py` 88 passed; `Tests/DB/` +
`Tests/ChaChaNotesDB/` 1046 passed / 1 skipped / 34 failed — byte-identical to the
pre-change baseline (the V33->V34 `duplicate column name: compaction_representation`
bug and its stale hardcoded-version assertions; not touched). `Tests/UI/
test_console_dictionary_send_integration.py`'s 2 failures were confirmed
pre-existing by running that file against a pristine copy of the base commit.
Not fixed here: `Tests/DB/test_sql_validation.py::test_no_missing_tables` stays red
on the three `console_*` tables a previous migration never allowlisted — the two
tables added here ARE allowlisted, so that failure is unchanged, not worsened.
<!-- SECTION:NOTES:END -->
