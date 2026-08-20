---
id: TASK-19045
title: >-
  Pin the ChaChaNotes index set: 84 of 94 named indexes are unreferenced by
  any test
status: To Do
assignee: []
created_date: '2026-08-20 08:40'
labels:
  - test-health
  - db
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-16840's close-out review constructed and disclosed a MUT-INDEX escape:
delete a `CREATE INDEX` from a ChaChaNotes migration step and nothing turns
red. The 16840 sweep cannot catch it by design — it compares a
chain-migrated DB against a fresh bootstrap, and both sides run the same
migration code, so a deterministic chain mutation is the identity on the
comparison (16840's own MUT-A honesty note in its Implementation Notes). No
absolute index census exists anywhere in Tests/.

Fresh census at dev `1bf7f234e` (live `CharactersRAGDB(":memory:")`,
`sqlite_master type='index'` minus autoindexes, isolated config): **94 named
indexes; only 10 of their names appear anywhere in Tests/; 84 are
unreferenced** — among them hot-path indexes like
`idx_messages_conversation_id_id`, `idx_msgs_conv_ts`,
`idx_conversations_workspace_id`, and the UNIQUE ledger-ordering
`idx_message_trajectory_conv_seq`. (16840's review counted 18-of-28 on the
narrower chain-created subset; the full gap is wider.) A dropped or renamed
index ships as a silent performance regression, and a lost UNIQUE index as a
silent integrity regression.

Precedent to mirror: `TestChachanotesValidTablesMatchesLiveSchema`
(`Tests/DB/test_sql_validation.py`, TASK-864 heritage) already does exactly
this census-against-live-schema pattern for tables.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A schema-census test asserts the full expected index-name set against a live, fully-migrated ChaChaNotes DB — red on any missing, renamed, or unexpected index
- [ ] #2 UNIQUE-ness is pinned for indexes where it is load-bearing (at minimum the trajectory `conv_seq` ledger index)
- [ ] #3 Mutation evidence: removing one `CREATE INDEX` from a migration step turns the census red (Edit-based restore, per repo lessons)
- [ ] #4 The expected set lives in one maintained place with drift guidance mirroring the VALID_TABLES pattern
<!-- AC:END -->
