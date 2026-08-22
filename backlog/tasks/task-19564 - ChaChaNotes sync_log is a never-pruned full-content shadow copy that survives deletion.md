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
* **12 retention triggers** (`sync_log_prune_<entity>` on UPDATE,
  `sync_log_prune_<entity>_hard` on DELETE, for messages / conversations /
  notes / character_cards / keywords / keyword_collections). The prefix is
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
  `_SYNC_LOG_RETENTION_SCOPES`, `prune_sync_log`, `delete_sync_log_entries`,
  `delete_sync_log_entries_before`
* `tldw_chatbook/DB/migrations/chachanotes_v44_to_v45_sync_log_retention.sql` (new)
* `Tests/DB/test_chachanotes_sync_log_retention.py` (new)
* `Tests/DB/test_chachanotes_sync_log_retention_migration.py` (new — carries the
  repo's exact schema-version pin, moved on from the v43→v44 file)
* `Tests/DB/test_chachanotes_sync_conflict_preservation_migration.py` — pin
  relaxed to `>= 44`
* `Tests/ChaChaNotesDB/test_chachanotes_db.py`,
  `Tests/ChaChaNotesDB/test_provider_continuation.py`,
  `Tests/Notes/test_note_import_executor.py` — three assertions that counted
  superseded `sync_log` history, rewritten to assert the frontier directly
