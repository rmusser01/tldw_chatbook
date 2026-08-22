---
id: TASK-19567
title: >-
  Soft-deleted messages remain searchable across conversations, and the FTS
  delete guard has no behavioural witness
status: Done
assignee: []
created_date: '2026-08-21 20:17'
updated_date: '2026-08-22 12:00'
labels:
  - db
  - privacy
  - testing
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review — Lane 3 **F7** (the live defect) and Lane 5
**MUT-14** (the coverage hole). Filed together because they are the same
guarantee failing in two layers. Both re-verified at this branch base.

**A — `search_messages_by_content` does not exclude messages of a deleted
conversation (CONFIRMED).** Soft-deleting a conversation leaves its messages at
`deleted = 0`. `ChaChaNotes_DB.py:11322-11333` filters `AND m.deleted = 0` but
**never joins `conversations`**:

```sql
FROM messages_fts fts
     JOIN messages m ON fts.rowid = m.rowid
WHERE fts.messages_fts MATCH ? AND m.deleted = 0
```

Its sibling method filters **both**. Honest reachability, carried through from
the lane: **not exploitable unscoped today** — the only live caller always
passes a `conversation_id` obtained from the already-filtered sibling. It is
one caller away from a leak, and the asymmetry between two sibling methods is
exactly the shape that produces that caller.

**B — the FTS soft-delete trigger guard is unobservable through every shipped
API (CONFIRMED, and sharper than "untested").** `ChaChaNotes_DB.py:563-571`
defines `messages_au` with `WHERE new.deleted = 0` at line 570. Lane 5 mutated
that guard away and **475 tests across every FTS-adjacent file still passed**.

Verification for this filing found the reason, and it matters for how the fix
is written: there **are** tests named for the behaviour
(`Tests/DB/test_search_conversations_fts.py:551
test_soft_deleted_message_does_not_match`), but **none can catch the mutation**,
because all four production `messages_fts` consumers redundantly re-filter at
query level — `ChaChaNotes_DB.py:8160-8169`, `:9308-9313`, `:11330-11333`,
`:12734-12736`. The only tests that query `messages_fts` **directly** are
`Tests/DB/test_chachanotes_provider_continuation_migration.py:147-149` and
`:229-231`, and they are a before/after snapshot **equality** check across a
migration — a uniform mutation is invisible on both sides.
`test_uses_messages_fts_match` only asserts the string `"messages_fts"` appears
in the source.

So the trigger guard is defence-in-depth with **no behavioural witness**: if it
regressed, the deleted content would sit in the FTS index and the only thing
standing between it and a search result would be four query-level filters that
each have to keep being right — including the one in (A) that is already
incomplete.

The sibling triggers (notes, conversations, character_cards, keywords,
keyword_collections) share the shape and are presumed to be in the same
position; confirm rather than assume.

## Acceptance Criteria

- [x] `search_messages_by_content` excludes messages whose conversation is
      soft-deleted, matching its sibling — pinned by a test that calls it
      **unscoped**, which is the shape that is one caller away
- [x] A test queries `messages_fts` **directly** (bypassing the query-level
      `deleted = 0` filters) and asserts a soft-deleted message's content is
      absent from the index
- [x] That test is mutation-checked: removing `WHERE new.deleted = 0` from
      `messages_au` makes it red — the current suite does not
- [x] The direct-index assertion is an absolute expectation, not a before/after
      equality snapshot, so a uniform regression cannot pass
- [x] The sibling triggers (notes, conversations, character_cards, keywords,
      keyword_collections) are checked for the same gap and given the same
      witness where they lack one
- [x] Any other FTS consumer added later inherits the guarantee — the coverage
      lives with the trigger, not scattered across each caller's `WHERE`

## Implementation Plan

1. Reproduce (A) unscoped and fix the asymmetry against the sibling.
2. Write the direct-`messages_fts` witness, then prove it by mutating the
   guard away.
3. Enumerate the sibling triggers from the LIVE schema rather than from the
   filing's list, and witness every one that turns out to share the shape.
4. Give the guarantee a home that a future FTS table inherits.

## Implementation Notes

### A — the sibling asymmetry

`search_messages_by_content` now joins `conversations` and filters
`c.deleted = 0`, matching `search_conversations_by_content`. Pinned by
`Tests/ChaChaNotesDB/test_chachanotes_db.py
::test_search_messages_by_content_unscoped_excludes_deleted_conversations`,
which calls it **unscoped** (the shape one caller away), asserts it now agrees
with the sibling it used to diverge from, and checks the scoped form does not
reopen the hole. Born red: at `da4e828af` the unscoped call returned the
deleted conversation's message body verbatim while the sibling returned
nothing.

### B — the witness, and the mutation proof

`Tests/DB/test_fts_soft_delete_index_witness.py` (13 tests). Every assertion
queries the FTS index **directly** — no consumer, so no query-level
`deleted = 0` can mask a trigger regression — and every one is an **absolute**
expectation (`== []` / `== [rowid]`), never a before/after equality snapshot.

Mutation proof (the deliverable, not the test's existence). Deleting
`WHERE new.deleted = 0` from the shipped `messages_au`:

* **new witness: 4 failed, 9 passed** — `..._drops_a_soft_deleted_message`,
  `..._drops_a_message_deleted_by_raw_update`,
  `..._restores_an_undeleted_message`, and the census guard.
* **pre-existing FTS-adjacent suite: 391 passed, 0 failed** —
  `Tests/DB/test_search_conversations_fts.py` (including
  `test_soft_deleted_message_does_not_match`, the test *named* for this
  behaviour), all of `Tests/ChaChaNotesDB/`,
  `test_chachanotes_provider_continuation_migration.py` and
  `test_provider_continuation_privacy.py`. Completely blind, reproducing Lane
  5's result exactly. That gap is now closed by this module and nothing else.

A second mutation, on a **sibling**: deleting the same guard from
`character_cards_au` turns 2 of the new witnesses red
(`..._drops_a_soft_deleted_card` + the census) while all **306** tests in
`Tests/ChaChaNotesDB/` stay green. The blindness was never specific to
`messages_au`.

The raw-`UPDATE` variant exists because `soft_delete_message` could grow an
explicit index-maintenance step and keep an API-driven test green with the
trigger broken.

### What the sibling sweep actually found

The filing named five siblings. Enumerating external-content FTS5 tables from
`sqlite_master` and checking which base tables carry `deleted` found **eight**:
the six named plus **`chat_dictionaries` and `world_books`**. All eight had a
correctly guarded insert half and none had a behavioural witness; all eight now
have one.

It also found a **live defect nobody had filed**. Five of the eight `*_au`
triggers guard their DELETE half with `WHERE old.deleted = 0` (`notes_au` was
repaired earlier for exactly this — see
`_ensure_notes_fts_update_trigger_handles_undelete`); three — `messages_au`,
`keyword_collections_au`, `world_books_au` — issued the FTS `'delete'`
unconditionally. Issuing it for a row that is not in an external-content index
corrupts that index.

`keyword_collections_au` was reachable through the shipped public API — at
`da4e828af`, `add_keyword_collection(name)` on a soft-deleted name goes through
`_add_generic_item`'s undelete UPDATE and raises `sqlite3.DatabaseError:
database disk image is malformed`. `messages_au` and `world_books_au` have no
undelete path today (every `UPDATE world_books` is `AND deleted = 0`), so they
were latent — fixed anyway, because latent here means "one restore API away",
and the census test now checks BOTH halves so the shape cannot come back.

All three are redefined in the v44→v45 migration, and the three indexes are
reset with FTS5 `'delete-all'` + a `WHERE deleted = 0` reinsert — deliberately
**not** `'rebuild'`, which re-derives from the base table with no `deleted`
filter and would re-index every tombstoned row, reintroducing the exact leak
the guard exists to prevent
(`test_upgrading_reindexes_only_live_rows_into_messages_fts`).

### Inheriting the guarantee

Two census tests keep the coverage with the trigger rather than with each
caller: one asserts the set of soft-deletable FTS-backed tables equals the
witnessed set (a new one is red until it gets a witness), the other asserts
every such table's AFTER UPDATE trigger carries **both** guards —
`new.deleted = 0` on the insert half (the leak guard) and `old.deleted = 0` on
the delete half (the corruption guard). The second is explicitly secondary — a
source-string assertion is what let `test_uses_messages_fts_match` pass while
the guard was mutated away.

### Modified/added files

* `tldw_chatbook/DB/ChaChaNotes_DB.py` — `search_messages_by_content` joins
  `conversations`
* `tldw_chatbook/DB/migrations/chachanotes_v44_to_v45_sync_log_retention.sql` —
  `messages_au` / `keyword_collections_au` redefinition + index reset
* `Tests/DB/test_fts_soft_delete_index_witness.py` (new)
* `Tests/ChaChaNotesDB/test_chachanotes_db.py` — the unscoped pin
* `Tests/DB/test_chachanotes_sync_log_retention_migration.py` — the FTS reset
  assertion
