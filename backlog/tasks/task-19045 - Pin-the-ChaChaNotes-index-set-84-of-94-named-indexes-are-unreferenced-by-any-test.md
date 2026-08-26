---
id: TASK-19045
title: >-
  Pin the ChaChaNotes index set: 84 of 94 named indexes are unreferenced by
  any test
status: Done
assignee: ['@claude']
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
- [x] #1 A schema-census test asserts the full expected index-name set against a live, fully-migrated ChaChaNotes DB — red on any missing, renamed, or unexpected index
- [x] #2 UNIQUE-ness is pinned for indexes where it is load-bearing (at minimum the trajectory `conv_seq` ledger index)
- [x] #3 Mutation evidence: removing one `CREATE INDEX` from a migration step turns the census red (Edit-based restore, per repo lessons)
- [x] #4 The expected set lives in one maintained place with drift guidance mirroring the VALID_TABLES pattern
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Census probe (isolated config) against a live `CharactersRAGDB(":memory:")`
   at the branch base: `sqlite_master type='index'` minus autoindexes, plus
   `PRAGMA index_list` (unique flag) and `PRAGMA index_info` (column tuples).
   (DONE — 94 named indexes, 10 UNIQUE, matching the filing's count.)
2. New `Tests/ChaChaNotesDB/test_index_census.py` holding the ONE maintained
   literal: `EXPECTED_CHACHANOTES_INDEXES`, name → IndexPin(table, unique,
   columns) — an explicit hand-written literal, NOT derived from the schema
   code it checks (a chain-derived oracle is the identity on chain mutations,
   16840's MUT-A honesty note). Tests, each with VALID_TABLES-style drift
   guidance in the failure message: missing-from-live (dropped/renamed index),
   unexpected-in-live (new index; message emits ready-to-paste entries),
   shape mismatch (table/unique/columns), and a dedicated UNIQUE pin for the
   trajectory ledger index `idx_message_trajectory_conv_seq` that stays red
   even if the big literal is mechanically "updated" on a downgrade.
3. Run the census against BOTH a fresh bootstrap and a chain-migrated DB
   (bootstrap at v4 via `historical_bootstrap.chachanotes_db_at_version`,
   reopen unpatched → full-chain replay) via a parametrized module fixture —
   catches stop/resume divergence the 16840 parity sweep normalizes away.
4. Mutation evidence (Edit-based restore): delete `CREATE INDEX ...
   idx_conversations_workspace_id` from `_MIGRATE_V12_TO_V13_SQL` (a
   migration-step index with ZERO Tests/ references); show the 16840 sweep
   stays GREEN under the mutation (the disclosed escape) and the new census
   goes RED naming the index; capture exact red text to a file; restore via
   Edit.
5. Point the "NOT CAUGHT HERE" honesty paragraph in
   `test_historical_bootstrap.py` at the new census (the index class is now
   pinned).
6. Gates: new file + `Tests/DB/test_sql_validation.py` +
   `Tests/ChaChaNotesDB/` green; repo-wide `--collect-only -q` sweep; ruff
   check/format on touched files; Implementation Notes; Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
New `Tests/ChaChaNotesDB/test_index_census.py` — the VALID_TABLES pattern
(TASK-864) applied to indexes. **The pin**: `EXPECTED_CHACHANOTES_INDEXES`,
one hand-maintained literal of all 94 named indexes (census at base
`25500ad87`, schema v40), each pinned as `IndexPin(table, unique, columns)`
— name set both directions, UNIQUE flag from `PRAGMA index_list`, key-column
tuple from `PRAGMA index_info`. Deliberately NOT derived from the schema code
(a chain-derived oracle is the identity on chain mutations — 16840's MUT-A
note; the module docstring says so). Partial-index WHERE clauses are
deliberately not pinned (formatting-sensitive; flag+columns are the durable
core — docstring discloses this).

**Census runs on TWO DBs** via a parametrized module fixture: fresh
bootstrap AND chain-migrated (genuine v4 via
`historical_bootstrap.chachanotes_db_at_version`, reopened unpatched →
full-chain replay), catching stop/resume divergence the 16840 parity sweep
normalizes away.

**AC#2 (UNIQUE decisions)**: the unique flag is pinned for ALL 94 (it is a
column of the same literal); the docstring enumerates why each of the 10
UNIQUE indexes is integrity-bearing (trajectory ledger ordering, message
identity/keyset pagination, notes file-path mapping, one-active-binding,
four RAG dedupe/identity constraints, two folder-tree invariants). The
trajectory `idx_message_trajectory_conv_seq` additionally gets a DEDICATED
test independent of the big literal, so mechanically "updating the literal"
on a red census cannot ride a UNIQUE downgrade through.

**AC#3 mutation evidence** (transcript in scratchpad `mutation_evidence.txt`,
quoted in the PR): deleted `CREATE INDEX ... idx_conversations_workspace_id`
from `_MIGRATE_V12_TO_V13_SQL` (a migration-step index with zero Tests/
references). Under the mutation the 16840 sweep
(`test_historical_bootstrap.py`) stayed GREEN 36/36 — the disclosed escape,
now demonstrated not just asserted — while the census went RED on BOTH
variants: `AssertionError: Pinned ChaChaNotes indexes are MISSING from the
live schema: ['idx_conversations_workspace_id']. ...` (message names the
literal to update for a deliberate change and the file to repair for an
accidental one). Restored via Edit; `git diff tldw_chatbook/` empty
afterwards.

**Also**: the "NOT CAUGHT HERE" honesty paragraph in
`test_historical_bootstrap.py` now points at the census as the cover for the
index class; the 16840 lessons entry got a closing sentence.

**Gates** (all read from files): census file 8/8 passed by name;
`Tests/ChaChaNotesDB/ + Tests/DB/test_sql_validation.py` 299 passed;
repo-wide `--collect-only -q` 51,478 collected, exit 0, no collection
errors; ruff check + format clean on both touched test files. Production
code untouched (diff is Tests/ + backlog/ only).
<!-- SECTION:NOTES:END -->
