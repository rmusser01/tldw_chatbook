# ADR-224: Sargable timestamp trio — migrate ordering columns to the canonical shape

Status: Accepted
Date: 2026-10-06
Task: TASK-34427 (plan draft called this "ADR-215"; the 215 slot was superseded by this 224 numbering)
Related: ADR-173 (canonical UTC timestamp storage format), TASK-32803.2 (flashcard due-boundary fix this ADR supersedes on the write side), task-32172 (julianday ordering note), ADR-147 (conversation archive index)

## Context

Three non-console query families cannot use an index because the columns they
filter and order on are wrapped in normalizing SQL functions:

1. `get_conversations_for_character` — the keyset cursor
   (`julianday(last_modified) < julianday(?)`) and
   `ORDER BY julianday(last_modified) DESC` defeat every index. The wrapping
   exists because `conversations.last_modified` holds MIXED formats: rows
   written by legacy versions carry SQLite's `CURRENT_TIMESTAMP` default
   (`YYYY-MM-DD HH:MM:SS`, space separator), while every current writer stamps
   the ADR-173 canonical shape (`YYYY-MM-DDTHH:MM:SS.mmmZ`). Raw text ordering
   mis-sorts the mix (`' '` 0x20 < `'T'` 0x54 on the same calendar date).
2. `get_due_flashcards` / `count_due_flashcards` —
   `datetime(next_review) <= datetime('now')` and
   `ORDER BY datetime(next_review) ASC` defeat `idx_flashcards_next_review`.
   TASK-32803.2 chose read-time normalization as the correctness fix and made
   `update_flashcard_review` write the space-separated shape to match the
   legacy rows; `next_review` therefore holds two legacy shapes in the wild —
   space-separated (post-32803.2 writes) and `.isoformat()`
   `YYYY-MM-DDTHH:MM:SS[.f]+00:00` (pre-32803.2 writes).
3. Character list (`list_character_cards_page` / `count_character_cards` /
   `list_character_cards`) — the `_USER_VISIBLE_CHARACTER`
   `json_extract(...)` visibility predicate has no index, and
   `ORDER BY name COLLATE NOCASE ASC` cannot use the BINARY-collation unique
   index on `character_cards.name`, so every browse page sorts a full scan.

ADR-173 already names the exit: *"a column with legacy rows … stays mixed
until it is separately migrated (backfill existing rows to canonical + change
the default)"*. This ADR is that migration for the two columns the three
families actually read.

## Decision

### Canonical format (adopted, not redefined)

The canonical stored shape is exactly ADR-173's
`YYYY-MM-DDTHH:MM:SS.mmmZ` — millisecond precision, `T` separator, `Z`
suffix, fixed 24-char width — as emitted by
`tldw_chatbook.Utils.timestamps.utc_now_iso()` and its in-module twin
`CharactersRAGDB._get_current_utc_timestamp_iso()`. (The task brief's
`%Y-%m-%dT%H:%M:%SZ` second-precision sketch was corrected to match the
existing package helper verbatim, per the plan's binding resolution; SQLite's
`strftime('%Y-%m-%dT%H:%M:%fZ', julianday(col))` produces exactly this shape.)

### One-time data migration (schema v78 → v79)

`DB/migrations/chachanotes_v79_...sql` (name at time of writing:
`chachanotes_v78_to_v79_sargable_timestamp_normalization.sql`) normalizes the
two ordering columns:

```sql
UPDATE conversations
   SET last_modified = strftime('%Y-%m-%dT%H:%M:%fZ', julianday(last_modified))
 WHERE last_modified NOT GLOB '????-??-??T??:??:??.???Z'
   AND julianday(last_modified) IS NOT NULL;
```

- Same statement for `flashcards.next_review` (plus `next_review IS NOT NULL`;
  NULL means "due now" and stays NULL).
- Idempotent by construction: already-canonical values fail the `GLOB` guard,
  so a re-run updates zero rows; the version bump is guarded
  (`WHERE ... version = 78`); indexes use `IF NOT EXISTS`.
- Unparseable garbage (`julianday(...) IS NULL`) is left untouched rather than
  silently written to NULL — a NOT NULL column must never be nulled by a
  format migration; such rows are reported here as a residual risk (none are
  produced by any known writer).
- Ordering-preserving: `julianday()` is monotonic in its input instant, so
  post-migration raw text order equals pre-migration `julianday()` order. A
  golden-ordering test pins this on a mixed-format fixture.
- The normalization `UPDATE` runs with TWO conversations triggers dropped and
  recreated verbatim (the v70→v71 precedent): `conversations_sync_update`
  (a bare UPDATE would enqueue a spurious `sync_log` "update" event per
  conversation — a storage-format change is not a content change) and
  `character_conversation_search_conversations_au` (a bare UPDATE would bump
  the search-projection revision and stamp CURRENT_TIMESTAMP dirty rows —
  the searchable content did not change, and the wall-clock stamps would
  make the migration non-deterministic). The FTS mirror triggers fire but
  re-insert identical titles (net zero).

Scope of normalization: ONLY `conversations.last_modified` and
`flashcards.next_review` — the columns the three families filter/order on.
`conversations.created_at`, `messages` timestamps, and flashcard
`last_review`/`updated_at` are payload-only for these families and are left
mixed (audit result; readers of those columns keep ADR-173's tolerant rules).

### Writer compliance (no `CURRENT_TIMESTAMP` for the normalized columns)

- `conversations.last_modified`: every writer already stamps
  `_get_current_utc_timestamp_iso()` — `add_conversation` (INSERT supplies
  the column explicitly), `update_conversation`, `set_conversation_archive_states`,
  `soft_delete_conversation`, `restore_conversation`. No change needed.
- The table's `DEFAULT CURRENT_TIMESTAMP` (base schema, v4) is a latent
  last-resort that only fires for an INSERT omitting the column; no such
  INSERT exists. Changing it requires a full table rebuild (the conversation
  table carries FTS + sync triggers and FKs), which this task's risk budget
  forbids. This is the documented "cannot safely convert" path: it is guarded
  by a canonical-format test on the public writer paths instead of being
  half-converted. Fresh databases replay the v78→v79 migration (empty tables)
  and inherit the same invariant.
- `flashcards.next_review`: `update_flashcard_review` switches from
  `strftime("%Y-%m-%d %H:%M:%S")` (TASK-32803.2's shape) to the canonical
  shape, superseding 32803.2's write-side choice. 32803.2's correctness goal
  (a due card reads as due) is preserved — first by the one-time migration
  for existing rows, then by all-canonical writes making the raw comparison a
  real time comparison again. `create_flashcard` leaves `next_review` NULL
  (= due now), unchanged. `last_review`/`updated_at` stay
  `CURRENT_TIMESTAMP` (payload-only; out of scope).

### Query rewrites and index strategy

- `get_conversations_for_character`: raw keyset
  (`last_modified < ? OR (last_modified = ? AND id < ?)`) and
  `ORDER BY last_modified DESC, id DESC`. New index
  `idx_conv_char_lm ON conversations(character_id, last_modified DESC, id DESC)`
  — the existing `idx_conv_char` (character_id alone) cannot serve the order,
  and `idx_conversations_archive` (archived-first) serves the wrong prefix.
  `deleted`/`scope_type`/`archived` stay per-row filters.
- `get_due_flashcards` / `count_due_flashcards`: raw
  `(next_review IS NULL OR next_review <= ?)` with the bound parameter
  produced in Python (`_get_current_utc_timestamp_iso()`), and raw
  `ORDER BY next_review ASC`. `idx_flashcards_next_review` is kept as-is; the
  planner resolves the OR into two index probes (`MULTI-INDEX OR`). A small
  temp b-tree still sorts the surviving due rows: an index cannot serve the
  order across the union of a NULL-predicate and a range, and the sort input
  is the due set, not the table — this is the honest optimum and the assertion
  pins "no `SCAN f`, index used, NULLs-first order preserved".
- Character cards: new partial expression index
  `idx_character_cards_visible_name ON character_cards(name COLLATE NOCASE)
  WHERE deleted = 0 AND json_extract(CASE WHEN json_valid(extensions) THEN
  extensions ELSE '{}' END, '$.actor_pack_persona_portrait_owner') IS NULL`.
  SQLite ≥ 3.9 accepts deterministic expressions (incl. `json_extract`) in
  partial-index WHERE; the shipped runtime is 3.49.1 and the test
  feature-detects it. This is preferred over a stored `is_user_visible`
  column (no write-path churn); the stored-column fallback was not needed.
  Browse pages/counters over the partial index scan user-visible cards in
  NOCASE order with no full table scan and no post-sort.

### Rollback / compatibility

- Older readers (julianday/datetime-wrapped SQL) read canonical values
  correctly — `julianday()` parses the ISO `T...Z` shape — so a v79 database
  opened by pre-v79 code degrades only to today's non-sargable (correct)
  behavior.
- Pre-v79 code writing a space-separated `next_review`/`last_modified` into a
  v79 database reintroduces a mixed row: the new raw ordering would mis-place
  it (the bug ADR-173 documents) until re-migration. Downgrade-with-writes is
  therefore unsupported for these two columns; read-only downgrades are safe.
- Keyset cursors (`before_last_modified`) are in-memory per session and
  post-migration values are canonical, so no stale-format cursor can reach the
  rewritten query within one process lifetime.

### Risks

- Mixed-format legacy data is the reason the migration exists; the golden
  test proves order equivalence on a fixture containing space-separated,
  `+00:00`-offset, and canonical shapes plus same-instant cross-format ties.
- The migration is one transaction in the chain (task-19553 rules): it either
  lands whole or leaves the DB at v78.
- Sync surface: no spurious `sync_log` rows (trigger dropped/recreated around
  the UPDATE); historical `sync_log` rows keep the format they were written
  with (append-only journal, not reinterpreted).
- `character_cards` gets its first expression index; the index census
  (`Tests/ChaChaNotesDB/test_index_census.py`) is updated as the deliberate
  schema-review act that file demands.
- **Backup/Recovery wiring (the fleet v78 precedent, followed exactly).** The
  restore validator compares a candidate's whole `sqlite_schema` catalog
  against frozen installed catalogs and migrates under a set-authorizer
  connection. This change extends that machinery the way the v77→v78 fleet
  bump did: `CHACHANOTES_V79_SCHEMAS` (+`CORE_SCHEMAS` head at 79, dictionary
  variant rebased, full v78 lineage accepted at version 78 with a
  `(78, 79)` stamp step), the shared-file Subscriptions hybrid tier
  (`_SUBSCRIPTIONS_SARGABLE_SCHEMAS`, stamp 79), the canvas frozen-catalog
  list, and a `sargable_migration` authorizer branch admitting exactly the
  events the installed `.sql` statements produce (normalization UPDATEs,
  two indexes + their internal REINDEX, the recreated sync trigger, and the
  trigger-sourced FTS/search-projection/sync-journal writes the UPDATEs
  compile). `Tests/DB/test_chachanotes_v79_sargable_migration.py` pins the
  restricted migration, the foreign-DDL rollback, and the stamp acceptance.

## Alternatives considered

- **Keep read-time normalization forever (status quo).** Correct but
  non-sargable: every page of a character's conversations and every due-card
  probe pays a full scan + sort.
- **Migrate to numeric epoch columns.** Ordering-safe but rewrites column
  types, every reader/writer, and the sync payloads; far outside this task's
  risk budget and rejected by ADR-173 for the same reason.
- **Stored `is_user_visible` column + triggers.** Only needed if SQLite
  rejected `json_extract` in the partial-index WHERE; the shipped 3.49.1
  accepts it, so the write path stays untouched.
- **Change the `conversations` table DEFAULT via rebuild.** Rejected as
  disproportionate risk for a default no INSERT relies on; documented above
  as the one consciously left site.

## Consequences

- Schema version 78 → 79; `_CURRENT_SCHEMA_VERSION` bumped; index census and
  the flashcard due-boundary contract tests updated to the canonical shape.
- Raw (sargable) comparisons become CORRECT for these columns because the
  all-canonical precondition of ADR-173's guarantee scope now holds for them.
- Any future writer of these two columns must emit the canonical shape; the
  writer-guard tests in `Tests/ChaChaNotesDB/test_sargable_timestamps.py`
  fail red on a divergent shape.
