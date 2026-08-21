---
id: TASK-19567
title: >-
  Soft-deleted messages remain searchable across conversations, and the FTS
  delete guard has no behavioural witness
status: To Do
assignee: []
created_date: '2026-08-21 20:17'
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

- [ ] `search_messages_by_content` excludes messages whose conversation is
      soft-deleted, matching its sibling — pinned by a test that calls it
      **unscoped**, which is the shape that is one caller away
- [ ] A test queries `messages_fts` **directly** (bypassing the query-level
      `deleted = 0` filters) and asserts a soft-deleted message's content is
      absent from the index
- [ ] That test is mutation-checked: removing `WHERE new.deleted = 0` from
      `messages_au` makes it red — the current suite does not
- [ ] The direct-index assertion is an absolute expectation, not a before/after
      equality snapshot, so a uniform regression cannot pass
- [ ] The sibling triggers (notes, conversations, character_cards, keywords,
      keyword_collections) are checked for the same gap and given the same
      witness where they lack one
- [ ] Any other FTS consumer added later inherits the guarantee — the coverage
      lives with the trigger, not scattered across each caller's `WHERE`
