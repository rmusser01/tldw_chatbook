---
id: TASK-19866
title: >-
  No guard prevents reintroducing a mid-body version short-circuit in migration tests
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - testing
  - test-integrity
  - database
priority: medium
dependencies:
  - TASK-19568
---

## Description

Source: **TASK-19568**'s reviewer, and independently recommended by Qodo on
that task's PR (#1940). Re-verified at `3605bd52d`.

A migration test that opens with `assert db.schema_version == N` before any
substantive assertion is not a test — it is a switch that turns the rest of the
body off. Once the schema moves past `N`, the equality fails (or the guard
skips), and the columns, indexes and row-survival assertions below it stop
running. The suite stays green because the file still "passes"; the migration
it was written to protect is simply no longer checked by anything.

TASK-19568 removed the offending assertions. It did not ship a guard, and its
own review is the proof that one is needed: the reviewer found the same shape
**already latent in the newest migration test file**
(`Tests/DB/test_chachanotes_sync_conflict_preservation_migration.py`), which
held three version equalities rather than the one the task had accounted for,
two of them positioned mid-body ahead of the `NEW_COLUMNS` and row-survival
assertions. A simulated v45 bump turned three of those files red where one
should have been. Nothing would have surfaced this until the next schema bump,
by which point the assertions had already been silently inert for a release
cycle.

Both the human reviewer and Qodo independently proposed the same remedy, which
is a reasonable signal that it is the right one: a meta-test over the migration
test modules that flags a literal version-equality assertion positioned before
the substantive assertions in its function body.

At `3605bd52d` there are 26 modules matching `Tests/**/test_*migration*.py` and
no guard of any kind over them.

## Acceptance Criteria

- [ ] A test fails when any module under `Tests/**/test_*migration*.py`
      contains a literal schema-version equality assertion positioned before
      the substantive assertions in the same function body
- [ ] The guard is structural (AST) rather than a text grep, so reformatting,
      line wrapping or an intervening comment does not evade it
- [ ] The guard is mutation-verified against at least three distinct evasion
      shapes — for example a version equality inside a helper called from the
      body, one wrapped in a `pytest.skip` guard, and one expressed as
      `!=`/`<=` rather than `==`
- [ ] The guard reports every offending file and line, not just the first
- [ ] Version assertions that are legitimately the *subject* of a test (a test
      whose whole purpose is that the version advanced) remain possible, and
      the mechanism by which they are distinguished is documented
- [ ] The guard is green against the current tree at the time it lands, with
      any allowlisted entry carrying a written reason

## Notes

The incident, kept with the rule: TASK-19568 was dispatched to remove exactly
this shape. Its reviewer then found the shape sitting in the newest migration
test file the task had just walked past — because a check scoped to "the file
we know about" structurally cannot see the file added last week. That is what a
guard is for, and it is why the acceptance criteria here demand mutation
evidence rather than a passing run.
