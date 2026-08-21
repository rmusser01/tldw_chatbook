---
id: TASK-19568
title: >-
  Schema pins merged red — two gates no longer gate, and an equality pin
  short-circuits the entire persona-visual migration test
status: To Do
assignee: []
created_date: '2026-08-21 20:18'
labels:
  - testing
  - db
  - architecture-gate
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 5 (test-suite health & guard efficacy)
— its **(b) masked gates** and **B3 stale schema pins**; Lane 3 **F3**
independently found the same `sql_validation` red, and the review controller
**re-confirmed it by running the test**. All numbers below were re-measured at
this branch base.

**One merge caused all of it.** `452d09b12` (persona-visual, PR #1882) added
the v40→v41 migration and updated neither pinned literal. The gates then fired
exactly as designed — and were **merged red anyway**, which is live proof of
their efficacy and of the process gap that let them be ignored (see
TASK-19572: no required status check exists on `dev`, so a red guard cannot
block a merge).

**Currently red at this base:**

| pin | state |
|---|---|
| `Tests/DB/test_sql_validation.py::test_no_missing_tables` (line 387) | RED — `assert not {'persona_visual_assets','persona_visual_bindings','persona_visual_pack_versions','persona_visual_packs'}`; whole file **1 failed / 22 passed** |
| `Tests/ChaChaNotesDB/test_index_census.py::test_no_unexpected_indexes` | RED under **both** parametrizations (`fresh_bootstrap`, `chain_migrated_from_v4`) — `idx_persona_visual_assets_version_key`, `idx_persona_visual_bindings_persona_active` |
| `Tests/DB/test_chachanotes_default_assistant_enrichment_migration.py:459` | RED — `assert 42 == 40` |
| `Tests/ChaChaNotesDB/test_persona_visual_migration.py:157` | RED — `assert 42 == 41` |

**The worst one is the last, and it is a self-inflicted coverage hole on the
exact migration that caused the others.** In
`test_persona_visual_migration.py`, the function
`test_real_v40_upgrade_installs_separate_persona_visual_schema` is defined
across lines 154-156 and **line 157 is the first statement of its body**:
`assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 41`. Because that assertion
fails immediately, **nothing below it ever runs** — not the v40→v41 migration
call at line 164, not the table census, not the five `EXPECTED_COLUMNS` checks,
not the foreign-key matrix, and **not the two index assertions at lines
184-185, which pin the very indexes the index census reports as unpinned**.
The persona-visual migration is, in practice, untested.

**Correction to the review's own summary, from this filing's verification:**
there is only **one** `== 40` pin, not two. The three ChaChaNotes equality pins
are `== 40` (red), `== 41` (red), `== 42` (green). A fourth hit,
`Tests/DB/test_media_db_schema_v6.py:19` (`== 6`), is the Media database — a
different class, correctly green, not part of this.

**Not in scope here:** `test_screen_size_ratchet` is the third masked gate but
is already tracked as **TASK-3070** (In Progress) and **TASK-3070.11**. Its
*unbudgeted* siblings are filed separately with the architecture findings.

The equality-pin pattern is the root cause worth fixing, not just the four
values: a test that asserts `_CURRENT_SCHEMA_VERSION == <literal>` guarantees a
red on the next migration, and a red at the top of a function body silently
deletes the rest of the test.

## Acceptance Criteria

- [ ] `sql_validation.VALID_TABLES` includes the four `persona_visual_*` tables
      and `test_no_missing_tables` is green
- [ ] The index census pins the two `persona_visual_*` indexes and is green
      under both parametrizations
- [ ] `test_real_v40_upgrade_installs_separate_persona_visual_schema` executes
      its body — the migration, the table census, the column checks, the index
      assertions and the foreign-key matrix all actually run and pass
- [ ] Schema-version pins stop using bare equality against
      `_CURRENT_SCHEMA_VERSION`: a migration test asserts the behaviour of
      *its own* migration (v40→v41 produces this schema), not the global
      current version
- [ ] A guard makes it hard to short-circuit a test body with a version
      assertion — e.g. version preconditions expressed as a skip/xfail marker
      or `>=`, so a bump cannot silently disable a whole test
- [ ] Adding a migration without updating the derived pins fails a check the
      author sees **before** merge (this is the dependency on TASK-19572)
- [ ] The full ChaChaNotes DB suite is green at the end, with counts recorded
