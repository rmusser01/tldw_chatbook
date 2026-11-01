---
id: TASK-33371
title: About 50 DB tests have drifted from production code (signatures, semantic-mutation
  guard, fixtures, stale pins)
status: To Do
created_date: 2026-09-28 20:12
dependencies:
- TASK-33370
labels:
- testing
- database
- tech-debt
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Besides the RecoveryRequired class (TASK-33370), about 52 DB and ChaChaNotesDB failure tracebacks on dev are real test/code drift:
- private._pin_sqlite_source() now requires reservation= and deadline= (ADR-125, 61a49de2e0), but Tests/DB/test_sqlite_source_pin_lifetime.py still calls the old signature.
- Raw-SQL fixtures are refused by the semantic-mutation guard ('semantic mutation authorization required for message delete/update, attachment insert', about 12).
- Migration fixtures are missing tables (console_trace_graph_epoch, note_links).
- Missing attributes: CharactersRAGDB._db_diagnostic_ref (6), is_memory_db (1), ChatScreen._dispatch_console_changed_files_worker (3).
- Stale pins: a schema version expectation of 54 against 73, and EXPECTED_CHACHANOTES_INDEXES missing live indexes.
- test_core_sqlite_owner_privacy.py fails 7 times with 'core owner resolved the selected database path', and one backup test with 'backup resolved its selected source or target path'. Those assertions guard a privacy contract, so they may be real product regressions rather than test drift. Found 2026-09-28 while verifying PERF-02 (PR #2887) and PERF-01 (PR #2888): the failures reproduce on the unchanged base commit 9cd9aad65f (dev 48019b1914 plus the audit docs commit), so they are pre-existing on dev.
<!-- SECTION:DESCRIPTION:END -->

Cross-reference (TASK-22280 closure, 2026-10-02, dev `e92b01515f`): the
`console_trace_graph_epoch` fixture gap was re-verified with exact tracebacks
(seeding dies in `soft_delete_message`/`_update_message_uncoordinated` ->
`_advance_semantic_graph_epoch`, ChaChaNotes_DB.py:13668; table declared at
`chachanotes_v55_to_v56_console_semantic_trace.sql:370`); candidate fix shapes
and guard/lesson pointers are recorded in TASK-22280's Implementation Notes.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each failing test is triaged as test drift (test fixed) or product defect (code fixed, or a task filed and named in this task's notes)
- [ ] #2 The core-owner and backup path-resolution privacy failures are explained with evidence before either side is changed
- [ ] #3 The affected files pass locally on dev, and the list of fixed tests is recorded in the implementation notes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
