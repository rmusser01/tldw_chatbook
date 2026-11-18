---
id: TASK-15663
title: 'Repair the local-marks v16 to v17 migration schema test (unowned dev red)'
status: Done
assignee:
  - '[rmusser01]'
created_date: '2026-08-11 21:30'
labels:
  - db
  - tests
  - baseline
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Chat/test_conversation_local_marks_service.py::test_local_marks_migrate_from_v16_to_v17_with_expected_schema` fails on a PRISTINE `origin/dev` detached checkout (1 failed, 13 passed), so it is not caused by any in-flight branch. It is not filed anywhere in the backlog and is plausibly the 13th red that PR #1500 ("repaired 12 of 13 dev baseline reds") left behind. A ChaChaNotes schema-migration red must be owned rather than left to become pre-existing noise that hides a real failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test passes on dev, or is deleted with a recorded reason if the contract it asserts no longer exists
- [x] #2 The diagnosis states whether the schema or the expectation was wrong, with evidence
- [x] #3 Tests/Chat runs with zero failures attributable to this file
<!-- AC:END -->

## Implementation Plan (added 2026-10-02, [rmusser01])

ADR required: no
ADR path: N/A
Reason: test-only verification and task close-out; no production schema, migration, or policy change.

1. Verify the premise at the assigned base (origin/dev tip ecc0a531c8): run the named test, then the whole containing file.
2. If green, attribute the fix: trace git history of the test file and its fixture helpers to the commit(s) that repaired it, and cross-check the owning task records (incl. possible duplicate filings).
3. Record the schema-vs-expectation diagnosis from the landed fix's own evidence; close as already-fixed upstream with reproduction commands and results.

## Implementation Notes (2026-10-02, [rmusser01])

**Classification: (a) already fixed upstream — closed with evidence, no code change.**

- **Premise verification at the assigned base (origin/dev tip `ecc0a531c8`)**: the named
  test is green.
  - `python -m pytest "Tests/Chat/test_conversation_local_marks_service.py::test_local_marks_migrate_from_v16_to_v17_with_expected_schema" -q`
    → 1 passed.
  - `python -m pytest Tests/Chat/test_conversation_local_marks_service.py -q`
    → 40 passed, 0 failed. A file that passes in full contributes zero failures to
  Tests/Chat, which is what AC #3 requires.
- **Who fixed it, and when (AC #2 diagnosis).** The red was repaired two days after this
  task was filed, by a later duplicate filing: TASK-16207 ("Repair local-marks V16
  migration fixture", created 2026-08-13 23:53 — this task was created 2026-08-11 21:30
  and predates it; recorded here per the duplicate-closeout discipline because 16207 is
  Done and stays untouched). Its commit `5300077fdb` ("test: repair local marks
  migration fixture") is the fixing change. **The expectation (the test fixture) was
  wrong, not the schema**: the registry-era synthetic V16 DB retained the
  V35→V36 `note_folders`/`note_folder_memberships` tables, so replaying the real
  migration chain from v16 failed at V35→V36 with duplicate-table errors. Evidence:
  TASK-16207's Implementation Notes record the reproduced RED and prove each of the two
  table drops necessary.
- **Why it has stayed fixed.** The fixture approach was since replaced wholesale:
  task-16840 (`86a0d315ab`) retired the rollback registry for
  `Tests/ChaChaNotesDB/historical_bootstrap.chachanotes_db_at_version`, which builds
  genuinely-historical DBs with the production chain itself (making the
  "artifact pre-exists its declaring migration" class impossible by construction);
  task-21441 (`9548de4165`) and task-32186 (`2bf446152d`) kept the bootstrap current
  through the v48 column and the v72→v73 `note_links` forward-write dependency. The
  current test pins exactly this bootstrap path and passes at dev tip.
- **Not owned by TASK-33371**: that open task's census (v55/v56 epoch writes, missing
  `console_trace_graph_epoch`/`note_links` fixtures, stale v54-vs-73 pins) covers other
  files; every test in this file is green at the base verified here.
- **ADR required: no.** Reason: no production schema, migration, or policy change —
  verification and task close-out only.
- **Modified files:** this task file only.
