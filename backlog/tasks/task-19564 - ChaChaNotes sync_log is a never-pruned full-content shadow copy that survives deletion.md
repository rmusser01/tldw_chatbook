---
id: TASK-19564
title: >-
  ChaChaNotes sync_log is a never-pruned full-content shadow copy that survives
  deletion
status: Done
assignee: []
created_date: '2026-08-21 20:14'
updated_date: '2026-08-22 12:00'
labels:
  - db
  - privacy
  - retention
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 3 (data layer & schema integrity) —
its **F5**, CONFIRMED LIVE by probe. Re-verified at this branch base.

The ChaChaNotes `sync_log` table is a **write-only, never-pruned, full-content
shadow copy** of the user's data:

- 35 triggers write the **complete row as JSON** into it.
- Both of its readers have **zero external callers** — nothing consumes it.
- There is **no `delete_sync_log_entries_before`** and **no `DELETE FROM
  sync_log`** anywhere for this database. (`Client_Media_DB_v2.py:2671,2690`
  and `Prompts_DB.py:4268` show the prune functions the Media and Prompts
  databases *do* have — ChaChaNotes simply never got one.)

The lane's probe: it soft-deleted a conversation, then found **the message body
still present in `sync_log`**.

Two consequences:

1. **Privacy.** "Delete" leaves the user's plaintext in the database
   indefinitely. A user who deletes a conversation has every reason to believe
   the content is gone; it is not. This is the same class of expectation-gap as
   the soft-delete search leak in TASK-19566.
2. **Size.** It roughly doubles on-disk size for message text, permanently.

## Acceptance Criteria

- [x] Deleting a conversation, message, note or character removes the
      corresponding content from `sync_log`, or `sync_log` stops storing full
      content in the first place
- [x] A pruning path exists for ChaChaNotes `sync_log`, matching what
      `Client_Media_DB_v2` already provides, and it actually runs rather than
      merely existing
- [x] The retention decision is explicit and recorded: if the log is genuinely
      unused (both readers have no external callers), retiring the content
      columns is the durable answer — do not keep writing a shadow copy nobody
      reads
- [x] A test reproduces the lane's probe: after a soft delete, the deleted
      content is not retrievable from `sync_log`
- [x] Existing users' databases are addressed — a fix that only helps new
      installs leaves the plaintext in place for everyone who already has it
- [x] The on-disk size effect is measured before and after on a realistic
      database

## Implementation Plan

1. Verify the "zero external callers" premise in BOTH directions before acting
   on it.
2. Derive which `sync_log` rows any reader can actually reach.
3. Bound the log to that frontier with triggers, so retention runs on every
   write rather than needing a caller.
4. Migrate existing databases with a one-time purge of the same set.
5. Add the `Client_Media_DB_v2` prune API parity ChaChaNotes never had.
6. Measure on-disk size before and after on a realistic corpus.

## Implementation Notes

### The retention decision, and why it is NOT retirement

**`sync_log` is alive, and the filing's premise is stale.** The AC pointed at
retiring the content columns on the strength of "both readers have zero
external callers". That is true of the two LEGACY readers
(`get_sync_log_entries`, `get_latest_sync_log_change_id` — test-only for this
database; `Prompt_Management/Prompts_Interop.py` calls the *Prompts* DB's
copies, not these). It is false for the database as a whole: **four more
readers** were added since that pattern was named, and three have live,
non-test callers:

| reader (`ChaChaNotes_DB.py`) | live caller |
| --- | --- |
| `read_committed_chat_sync_intent` | `Chat/console_chat_store.py:6548` (`ensure_provider_continuation_durable`) → `:6741`, on every provider-continuation checkpoint |
| `read_committed_chat_delete_intent` | via `list_current_committed_chat_sync_intents` |
| `list_current_committed_chat_sync_intents` | `Chat/console_chat_store.py:1431` (`_reconcile_restored_chat_sync_intents`), on every conversation restore with Sync v2 configured |
| `_previous_committed_chat_payload_hash` | internal to the two intent readers |

Each compares the sync_log payload to the live `messages` row **field by
field** (`intent_payload != expected_intent`) — the payload IS the proof that
the exact row was committed. Dropping `content` from it would make every
comparison fail, and the failure is silent-then-loud: Sync v2 would quietly
stop producing envelopes, and `ensure_provider_continuation_durable` **raises**
on a `None` read, so every continuation checkpoint would become a hard error.
`Tests/DB/test_chachanotes_sync_log_retention.py
::test_retention_does_not_break_the_committed_intent_readers` is the standing
guard on that trade.

**Demonstrated, not argued.** A probe stripped the `content` key out of the
`sync_log` payloads — exactly what retiring the column would do — and re-read
through the shipped API:

```
DIRECTION 1 (content present): reader returns a RECORD  content='the exact body'
DIRECTION 2 (content retired):  reader returns None
DIRECTION 2: restore-reconcile intents = []
```

`ensure_provider_continuation_durable` raises on that `None`, and note it
reaches the reader **before** consulting `sync_v2_server_profile_id` — so this
breaks in the DEFAULT configuration, with sync never set up, not only for users
who enabled it. That is the difference between dormant-but-intended and dead.

So the answer is not "stop writing content", it is **"stop keeping content
nobody can reach"**. A row is reachable only through a JOIN to its live entity
row on `entity_id` AND `version`:

* messages, live → `{v, v-1}` (`v-1` feeds the base-hash lookup)
* messages, tombstoned → `{v}` only — the tombstone carries no content
* every other entity → `{v}` only — nothing reads them
* orphans (entity row gone) → nothing

### What shipped

* **Schema v45** (`_CURRENT_SCHEMA_VERSION` 44 → 45) +
  `DB/migrations/chachanotes_v44_to_v45_sync_log_retention.sql`, run through
  `_execute_migration_statements` inside `self.transaction()` per task-19553's
  atomicity convention. Bare `CREATE TRIGGER` statements so
  `_drop_superseded_trigger` makes the step re-enterable.

  Atomicity is not just asserted at source level. This step DELETEs user rows,
  so a half-commit would destroy content while leaving the stamp at 44 and the
  step re-entering forever. `test_a_failure_mid_step_rewinds_to_v44_with_
  nothing_applied` poisons a statement appended AFTER every purge DELETE and
  requires the stamp back at 44, zero triggers created and zero rows removed,
  then migrates the same file cleanly once the poison is gone. Mutation-checked
  the same way as the FTS guard: swapping the runner to `cursor.executescript`
  reds it (and the repo's source-level pin).
* **18 retention triggers**, covering all nine entities the schema writes
  `sync_log` rows for, under two rules. Twelve versioned
  (`sync_log_prune_<entity>` on UPDATE, `sync_log_prune_<entity>_hard` on
  DELETE, for messages / conversations / notes / character_cards / keywords /
  keyword_collections) and six latest-only (`AFTER INSERT ON sync_log` plus a
  hard-delete companion, for chat_dictionaries / world_books /
  world_book_entries — see "The three uncovered writers" below for why the
  version rule cannot bound those three). The prefix is
  deliberate: the `<entity>_sync_%` namespace belongs to the four triggers that
  WRITE the log, and three tests assert its membership exactly — `_` is a
  single-character wildcard in SQL LIKE, so `conversations_sync_log_prune`
  would have squatted in it (and did, until
  `test_local_project_context_is_excluded_from_conversation_sync_triggers`
  caught it). They only ever
  delete rows at versions STRICTLY BELOW the frontier, so they cannot perturb
  `list_current_committed_chat_sync_intents`'s `1 = (SELECT COUNT(*) …)`
  single-intent check. This is the "it actually runs" half — no scheduler, no
  caller to forget.
* **One-time purge** in the same migration for databases that already have the
  backlog, plus `prune_sync_log()` as the same sweep on demand, and
  `delete_sync_log_entries()` / `delete_sync_log_entries_before()` for parity
  with `Client_Media_DB_v2` (whose own `delete_sync_log_entries_before` has
  zero production callers — this one does not depend on being called).

### Measured on-disk effect

Realistic corpus: 60 conversations × 40 messages (2,400 messages, 2.86 MB of
message text) with 30% of messages edited 1–4×, 10% soft-deleted, plus 120
notes edited 3× each. Same seed both sides; sizes post-`VACUUM`.

| | file | sync_log rows | sync_log payload |
| --- | --- | --- | --- |
| v44 (shipped) | **19,017,728 B** | 4,940 | 8,406,302 B |
| v45 (fresh) | **14,221,312 B** (−25.2%) | 3,217 | 4,912,435 B (−41.6%) |
| v44 file upgraded → v45 | **14,200,832 B** (−25.3%) | 3,217 | 4,912,435 B (−41.6%) |

`sync_log` payload as a multiple of the message text it shadows: **2.94× →
1.72×**. The upgraded file lands on exactly the same `sync_log` state as a
fresh v45, which is the evidence for the existing-databases AC. Migration cost
on that database: **0.06 s**, freeing 959 × 4 KB = 3.93 MB inside the file
(`VACUUM` is what returns it to the filesystem; the migration does not run one).

Write cost of having 12 more triggers fire on every mutation, same corpus:
**1.87 s → 1.94 s (+4.1%)**. The prune is a single indexed `DELETE` per write
against `idx_sync_log_entity (entity, entity_id)`.

The table above was measured before the six latest-only triggers were added.
Three of those six are `AFTER INSERT ON sync_log`, so their `WHEN` is evaluated
on **every** log insert including the six versioned entities' — the case worth
measuring separately, since the rest of the corpus never touches a dictionary
or a world book. On a 800-message / 30%-edited corpus, best of three runs each:
**0.371 s without them → 0.374 s with them (+0.8%)**, i.e. three literal
string comparisons per emitted row and no extra statement. The size figures
above are unaffected (that corpus writes none of the three entities).

### Known residue (deliberate, not an oversight)

Soft-deleting a **conversation** does not soft-delete its messages — they stay
`deleted = 0` and come back on restore — so each message's single frontier row
is retained, exactly as `messages.content` itself is. After this change
`sync_log` never holds message text that `messages` does not; it is a bounded
frontier, not an unbounded shadow copy. Erasing the plaintext for LIVE rows too
requires the payload to carry a content HASH instead, which is a format change
to a live sync proof (and `_previous_committed_chat_payload_hash` reconstructs
a canonical hash from the stored content, so it would need a precomputed hash
plus `role` in the payload). **Recommended as a separate follow-up task, not
yet filed** — it needs an owner call on changing a live sync proof's format,
and a task id assigned against origin/dev.

### The three uncovered writers — found in review, now CLOSED (2026-08-22)

The first cut covered the six entities the filing named. `sync_log` is written
by **nine**. `chat_dictionaries`, `world_books` and `world_book_entries` had
sync-emitting triggers and **no retention trigger and no purge**, so for them
the original defect was untouched. This branch's own review recorded that as a
documented gap; **Qodo's review of PR #1974 reached the same finding
independently**, which is a strong signal it should not ship as a gap at all.
It does not: the rule is extended here rather than deferred.

Reproduced at v45 before the fix, through the shipped public APIs:

| sequence (public API path) | result |
| --- | --- |
| soft-delete a chat dictionary | `name`/`description`/`file_path` still in `sync_log` (3 rows after 2 edits) |
| soft-delete a world book | `name`/`description` still in `sync_log` |
| hard-delete a world-book entry (`world_book_manager.delete_world_book_entry`, wired to Personas ▸ entry delete at `UI/Screens/personas_screen.py:5221`) | the entry's full `keys` + `content` survive as an orphan, forever |
| 4 edits of a world-book entry | 4/4 old bodies retained (unbounded growth continues) |

**Why the six's rule does not extend to them.** Two findings from probing a
live v45 database, both of which invalidate a naive `version < NEW.version`
trigger:

* `world_book_entries` has **no `version` column and no `deleted` column**.
  Every one of its sync rows is written at the *literal* version 1 (read the
  emitter: `… , 1, json_object(…)`), so a version predicate is not merely weak
  there, it is inert — it can never be true. Its only delete path is a hard
  `DELETE`, which orphans every content row it wrote.
* `chat_dictionaries` has a `last_modified` timestamp trigger whose nested
  `UPDATE` re-fires the `sync_update` emitter. With `recursive_triggers` off
  (the default) a trigger cannot re-enter *itself*, but it does fire the
  *other* triggers — so when the emitted `last_modified` differs from the one
  the outer statement wrote, a **full-payload `update` row lands at the
  tombstone's own version**. Directly observed, prune triggers removed:
  `[(cid 2,'create',v1,body=True), (cid 3,'update',v2,body=True), (cid
  4,'delete',v2,body=False)]` — the deleted dictionary's plaintext sitting at
  the same version as its tombstone, invisible to `version < NEW.version`.

**The rule that shipped.** A second family, `_SYNC_LOG_LATEST_ONLY_SCOPES`,
anchored to the *log row* rather than to a version — six new triggers, all
`AFTER INSERT ON sync_log … WHEN NEW.entity = '<E>'` plus an `AFTER DELETE`
companion on each base table. At most **one** content-bearing row survives per
entity, the most recently emitted, and only while the entity is live;
content-free `delete` tombstones are kept as the delete proof. For the two
versioned entities the rule is "the six's rule, PLUS: while soft-deleted, no
content-bearing row survives at all".

**Order-independence — the argument, and the experiment.** SQLite leaves the
firing order of same-kind triggers undefined and a single soft delete really
does fire two emitters, so the rule must not depend on which fires first:

1. It fires once per *emission*, whoever emitted it, and its predicate reads
   only (a) the base table's state — already final for the statement, in any
   `AFTER` trigger — and (b) `change_id`, which is the table maximum at insert
   time.
2. Each firing re-establishes the same post-condition for that entity:
   content-bearing rows = `{the row just inserted}` if the entity is live,
   `{}` otherwise. The last emission of the statement therefore fixes the
   final state, and *which* trigger was last cannot change it: if the entity
   ends deleted the answer is `{}` either way; if it ends live the two
   candidate payloads are the same row rendered twice.
3. The `world_book_entries` hard-delete companion deliberately excludes
   `operation = 'delete'`. Deleting everything there *would* be
   order-dependent — fired before the sibling tombstone emitter it removes
   nothing, fired after it removes the tombstone.

The experiment, not just the argument: every scenario was re-run under six
permutations of the emitters' creation order, with a **control** proving the
permutation really reaches the firing order (without prune triggers the same
soft delete emits `update@cid3, delete@cid4` in one order and `delete@cid3,
update@cid4` in the other). All six permutations produced byte-identical
retained content. That control is now
`test_latest_only_retention_is_independent_of_trigger_firing_order`, which
asserts the two orders *differ* without retention and *agree* with it — so it
cannot pass vacuously.

**The census, so this cannot recur.** `test_every_sync_log_writer_has_a_
retention_scope` parses `sqlite_master` for triggers containing
`INSERT INTO sync_log`, extracts each entity literal, and asserts
`covered == writers` **both directions**; a sibling asserts every writer has
both of its `sync_log_prune_*` triggers. There is deliberately **no
allowlist** — all nine are covered, so an exemption row would have nothing to
hold. Bite-proof: dropping `world_book_entries` from the scope tuple reds it
with `sync_log writers with no retention rule: ['world_book_entries']`;
renaming one trigger in the migration reds both the runtime and the schema
census.

**Identifier validation (Qodo finding 2).** `prune_sync_log()` interpolated
`{table}` / `{id_expr}` / `{floor_expr}` straight into an f-string. Now only
two fragments are identifiers at all — the table and its id column — and the
scope tuples carry those as *names*, not expressions, routed through
`sql_validation.validate_table_name` / `validate_column_name` and
`escape_identifier` by `_sync_log_scope_identifiers()` before the sweep's
`try:` (so a rejected scope surfaces as itself, not as a generic "failed to
prune"). Everything else — the version floor, the liveness clause, the
tombstone exclusion — is a **fixed SQL literal selected by a `bool`**, so
there is no string for a caller to influence; that is the structural half of
the answer for the fragments an identifier checker cannot validate.
`validate_column_name` fails *closed* for a table with no `VALID_COLUMNS`
entry, so the three new tables were registered there and pinned against a live
schema by `test_sync_log_latest_only_table_columns_are_live`.

**One pre-existing defect deliberately NOT fixed here.** Hard-deleting a
`world_books` row cascades to `world_book_entries`, whose `sync_delete`
emitter reads `(SELECT client_id FROM world_books WHERE id =
OLD.world_book_id)` — already gone during the cascade — and raises `NOT NULL
constraint failed: sync_log.client_id`. No shipped path hard-deletes a world
book (`delete_world_book` is a soft delete), and retention neither causes nor
worsens it. Recorded rather than fixed inside a retention change.

Also worth recording: `get_sync_log_entries(since_change_id=…)` has **no**
version filter — it is a change-feed API, and after v45 it returns a frontier
snapshot rather than a complete log. It has zero ChaChaNotes production callers
today (the production hits belong to the Prompts DB's same-named method), so
nothing breaks; but anything built on it later must be written knowing the feed
is lossy by design.

### Trap worth carrying forward

The retention triggers were first named `<entity>_sync_log_prune`. That squats
in the `<entity>_sync_%` namespace, and **`_` is a single-character wildcard in
SQL `LIKE`** — so `conversations_sync_log_prune` matches
`LIKE 'conversations_sync_%'`. Three tests assert that namespace's membership
*exactly* as a design invariant ("these four triggers, and only these, write
the sync log"). Only one of them was red, because the other two run against
pre-migration historical databases where the new triggers do not exist yet —
the collision was two-thirds latent. Renamed to `sync_log_prune_<entity>`,
which is also the honest name: retention is a different concern from emission.

### Modified/added files

* `tldw_chatbook/DB/ChaChaNotes_DB.py` — version bump, `_migrate_from_v44_to_v45`,
  `_SYNC_LOG_RETENTION_SCOPES`, `_SYNC_LOG_LATEST_ONLY_SCOPES`,
  `_sync_log_scope_identifiers`, `prune_sync_log`, `delete_sync_log_entries`,
  `delete_sync_log_entries_before`
* `tldw_chatbook/DB/migrations/chachanotes_v44_to_v45_sync_log_retention.sql` (new)
* `tldw_chatbook/DB/sql_validation.py` — `VALID_COLUMNS` entries for
  `chat_dictionaries` / `world_books` / `world_book_entries`, without which
  `validate_column_name` fails closed for the retention sweep
* `Tests/DB/test_chachanotes_sync_log_retention.py` (new)
* `Tests/DB/test_chachanotes_sync_log_retention_migration.py` (new — carries the
  repo's exact schema-version pin, moved on from the v43→v44 file)
* `Tests/DB/test_sql_validation.py` — the three new column sets pinned against a
  live migrated database
* `Tests/DB/test_chachanotes_sync_conflict_preservation_migration.py` — pin
  relaxed to `>= 44`
* `Tests/ChaChaNotesDB/test_chachanotes_db.py`,
  `Tests/ChaChaNotesDB/test_provider_continuation.py`,
  `Tests/Notes/test_note_import_executor.py` — three assertions that counted
  superseded `sync_log` history, rewritten to assert the frontier directly

### Verification (final state)

`Tests/DB` + `Tests/ChaChaNotesDB` + `Tests/Sync_Interop`: **1723 passed, 1
skipped** (1710 → +13 new). `Tests/Notes`: **2847 passed, 5 skipped**.
`Tests/Character_Chat` + `Tests/Architecture`: 1162 passed, 1 skipped, 4 failed
— all four in `Tests/Architecture` and about `UI/Screens/chat_screen.py`, a
file this branch does not touch (pre-existing dev reds). Repo-wide
`--collect-only -q`: **56182 tests collected**, with
`Tests/UI/test_library_file_notes_workspace.py` ignored — it errors at
collection (`function uses no argument 'push_phase'`) on a file identical to
the merge base, last touched 2026-08-20 by an unrelated commit.
