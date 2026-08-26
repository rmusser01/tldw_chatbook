---
id: TASK-22225
title: 'v48 policy seeding: skip deleted conversations'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-26 06:13'
labels:
  - database
  - migration
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22225).

`DB/ChaChaNotes_DB.py:5953-5970`: the v48 bump seeds
`console_conversation_library_policy` with one row per conversation via
`INSERT ... SELECT id FROM conversations` with no `WHERE deleted = 0` — O(all
conversations ever) inserts inside the boot version-bump transaction, permanently storing
rows for tombstoned conversations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The seeding migration (current version at fix time) excludes deleted conversations; a fresh-migration test proves it
- [x] #2 Existing over-seeded rows are cleaned or explicitly documented as inert
- [x] #3 Migration remains self-contained (the TASK-21441 lesson)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish what a policy row for a soft-deleted conversation actually does today (read path, both write paths, turn-commit path) before choosing a cleanup story.
2. Decide the shipped-migration approach: fix the v47->v48 seed for databases that have NOT yet migrated AND add a forward v49->v50 step that removes the over-seeded rows from databases that already ran v48, so both populations converge in one open.
3. Red-first tests in a new Tests/DB/test_chachanotes_v50_console_policy_tombstone_cleanup.py: fresh v47->current over a DB with live+deleted conversations seeds only the live one; a DB migrated by the SHIPPED v48 seed (replayed verbatim) is cleaned by v50; re-running v50 is idempotent; live and user-authored rows survive; a failure inside the step rolls back to entry state.
4. Implement: WHERE deleted = 0 in _seed_console_library_policy_rows; chachanotes_v49_to_v50_console_policy_tombstone_cleanup.sql + _migrate_from_v49_to_v50 + registry entry; bump _CURRENT_SCHEMA_VERSION to 50; move the version pin out of the v49 test file per the newest-migration-owns-the-pin convention.
5. Update the v48 migration tests/docstrings that assert the old seeding contract, and amend ADR-079's 'active and soft-deleted conversations' sentence.
6. Measure the seeding cost at a realistic conversation count (before/after row counts + wall).
7. Targeted suites + --collect-only sweep, tee everything, counts read from the tees; ./scripts/preflight.sh; mutation-test both halves; failure walk for partial apply.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both halves shipped: `_seed_console_library_policy_rows` gained `WHERE deleted = 0`,
and a new v49→v50 step (`chachanotes_v49_to_v50_console_policy_tombstone_cleanup.sql`)
deletes every policy row with no live conversation behind it.

**Why editing the already-shipped v48 is the honest choice.** Editing an applied
step can only change the outcome for a database that has not yet reached it, so the
edit alone guarantees permanent divergence between the two populations — that is the
hazard, not the edit. The forward step removes it: a profile that already ran the
shipped seed and one that upgrades now reach the same rows inside one
`_initialize_schema` open. Doing it *only* forward would have been worse for the
finding's own subject: every not-yet-upgraded user would still pay the full
O(all conversations) insert and then a delete on top. The convergence is asserted
rather than argued — the test replays the shipped seed verbatim under `patch.object`
and compares the two profiles row for row.

**What a policy row for a deleted conversation actually did: nothing observable.**
`ConsoleLibraryPolicyRepository._POLICY_SELECT` joins `conversations` and
`_result_from_row` fail-closes to Never/Blocked whenever `conversation_deleted != 0`;
`insert` and `compare_and_swap` both return `MISSING_CONVERSATION` for a missing or
deleted conversation; `commit_durable_turn` raises "Durable conversation is
unavailable." before it reads policy; and there is no conversation-undelete path in
the application (only media has one). So this is a durable-state and storage repair,
not a correctness one — and that read predicate is exactly what the cleanup deletes.
Removing the rows is safe because "conversation with no policy row" is an ordinary
state: `add_conversation` has never written one, and `ConsoleLibraryPolicyCoordinator
.save` inserts revision one on demand for a live conversation that lacks it.

**Measured** (5 runs each, v47 profile with 3,000 live + 2,000 soft-deleted
conversations, A/B against base `f0e896122` by checking the one production file in
and out of the tree):

| | policy rows | file after | one-time open |
| --- | --- | --- | --- |
| not yet at v48, before | 5,000 (2,000 for tombstones) | 7,819,264 B | 14.0–15.4 ms |
| not yet at v48, after | 3,000 (0) | 7,704,576 B | 16.2–17.2 ms |
| already at v49, after | 5,000 → 3,000 | unchanged (pages freed, no VACUUM) | 9.8–12.5 ms |

Honest reading: this is **not** a boot-time win. The new step costs slightly more
than the 2,000 inserts it saves, once; what it buys is 2,000 fewer permanent rows and
a 114,688 B smaller file on the fresh-upgrade path, and it retires policy for
conversations the user deleted.

**The mutation test found a hole in the first test set.** Dropping `WHERE deleted = 0`
left all 33 migration tests green: every whole-chain assertion runs *after* v50, which
deletes exactly the rows a broken seed writes, so the end state is identical and only
the boot cost returns. `test_the_v48_seed_itself_never_writes_a_tombstone_row` stops
the chain AT 48 via `chachanotes_db_at_version(path, 48)` — the only point where the
seed's own output is observable — and it reds under that mutation. Neutering the
cleanup reds 4 tests including the convergence one.

Self-contained (AC #3): the step reads only the database, and
`Tests/Packaging/test_installed_distribution.py::test_installed_distribution_migrates
_v35_database_to_current` (the README's canary) passes for both wheel and sdist,
along with the two tests that assert every migration script ships.

Files: `DB/ChaChaNotes_DB.py` (seed predicate, new step, registry, version 50, v48
docstring), `DB/migrations/chachanotes_v49_to_v50_console_policy_tombstone_cleanup
.sql`, `DB/migrations/README.md` (new "Editing a migration that has already shipped"
section), `Tests/DB/test_chachanotes_v50_console_policy_tombstone_cleanup.py` (new,
holds the exact schema-version pin), `Tests/DB/test_chachanotes_console_library_policy
_migration.py` + `..._v49_messages_fts_update_scope.py` (contract/pin updates),
`backlog/decisions/079-console-library-conversation-authority.md` (amendment),
`Docs/security/production-diagnostic-inventory.json` (+3 calls — the new step's three
log lines, verbatim mirrors of V48→V49's, reviewed statement by statement before
regenerating).

Test counts from tees: 9 new tests green; `Tests/DB/` + `Tests/ChaChaNotesDB/`
1,832 passed / 1 failed (`test_image_data_integrity`, missing numpy);
`Tests/Packaging/` 113 passed after installing setuptools+pip into the worktree venv,
remaining failures all environmental (uv venv lives inside the checkout, so an
installed-tree probe sees its own `site-packages` under the source root); preflight
all green; `--collect-only` 59,608 collected, 28 errors all missing optional deps.
<!-- SECTION:NOTES:END -->
