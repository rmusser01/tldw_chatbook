---
id: TASK-20971
title: >-
  The chachanotes VALID_TABLES guard went red again within a day of being
  repaired
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - bug
  - database
  - testing
  - test-integrity
  - process
priority: high
dependencies:
  - TASK-19568
  - TASK-19057
---

## Description

Source: found on a clean `origin/dev` worktree at `684c6aba4` while renumbering
TASK-19564's migration after the schema-version collision with TASK-19057.

`Tests/DB/test_sql_validation.py::TestChachanotesValidTablesMatchesLiveSchema::
test_no_missing_tables` is red on `dev`:

> `AssertionError: Live schema has tables not in VALID_TABLES['chachanotes']:
> ['actor_pack_persona_intents', 'actor_portable_identities']. Add them (or
> document a deliberate exclusion) in tldw_chatbook/DB/sql_validation.py.`

TASK-19057's v44→v45 migration created two tables and did not add them to the
hand-maintained literal in `DB/sql_validation.py`.

**The drift itself is minor. The recurrence interval is the finding.** This is
the same pin TASK-19568 repaired — that task existed *because* the entry had
gone stale and a red gate stops gating. Its repair merged at `aaec11812`,
2026-08-22 00:16 -0700. TASK-19057 merged at `2fe6ca20f`, 2026-08-22 14:51
-0700 and broke it again. **Fourteen and a half hours.** A one-shot literal
update demonstrably does not hold, and this programme has already paid for a
masked gate once: TASK-19191's per-row inventory review cleared 13 red gate
tests and surfaced three privacy findings that had been sitting behind them,
and TASK-19044's stale version pin had been masking a shipped can't-migrate-past-v39
packaging bug.

Scope note, measured: the **index** census is unaffected —
`idx_actor_pack_persona_intents_state` is already pinned
(`Tests/ChaChaNotesDB/test_index_census.py:93`), and
`Tests/ChaChaNotesDB/test_actor_pack_migration.py` covers the new tables'
migration. It is specifically the `VALID_TABLES` tables literal that drifted,
because nothing connects adding a table to updating it except a human
remembering.

The related TASK-19867 covers the `media` and `prompts` entries of the same
allowlist, which have drifted further and have no guard at all. This task is
about the entry that *does* have a guard and went red anyway.

## Acceptance Criteria

- [ ] `Tests/DB/test_sql_validation.py::TestChachanotesValidTablesMatchesLiveSchema`
      is green on `dev`
- [ ] `VALID_TABLES['chachanotes']` matches the tables a freshly initialized
      ChaChaNotes database actually contains
- [ ] A mechanism exists that makes the *next* table-adding migration fail
      loudly at authoring time rather than after merge — the outcome required is
      that the literal cannot silently fall behind the schema again, not that it
      is correct today
- [ ] The chosen mechanism is proven to bite: adding a table without updating
      the allowlist is shown to fail, and the proof does not depend on the same
      hand-maintained list it guards
- [ ] The recurrence is recorded where a future migration author will encounter
      it, so the cost of this class of drift is not re-learned a third time

## Notes

Not filed as "update the literal". Twice-repaired-in-a-day is evidence that the
repair shape is wrong, and the acceptance criteria are written to ask for
whatever stops recurrence rather than for another correct-on-the-day list.

This is residue of the TASK-19057 merge, not of the retention work that found
it: `VALID_TABLES` is byte-identical to `dev`'s on TASK-19564's branch, and that
branch's migration contains zero `CREATE TABLE` statements.
