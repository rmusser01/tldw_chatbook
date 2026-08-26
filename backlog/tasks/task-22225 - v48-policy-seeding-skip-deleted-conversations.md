---
id: TASK-22225
title: 'v48 policy seeding: skip deleted conversations'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-26 04:35'
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
- [ ] #1 The seeding migration (current version at fix time) excludes deleted conversations; a fresh-migration test proves it
- [ ] #2 Existing over-seeded rows are cleaned or explicitly documented as inert
- [ ] #3 Migration remains self-contained (the TASK-21441 lesson)
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
