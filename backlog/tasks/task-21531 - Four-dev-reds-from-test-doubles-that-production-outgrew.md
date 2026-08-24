---
id: TASK-21531
title: >-
  Four dev reds from test doubles that production outgrew - Notes library and
  lasting-sync flow
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - testing
  - dev-red
  - stale-doubles
priority: medium
---

## Description

Four tests are red on pristine dev because their doubles were never updated when the
production signatures they stand in for changed. They are not testing anything real in this
state, and they sit in the red baseline every branch inherits. Same class as the test-suite
programme's "stale test doubles production has outgrown" finding.

## Acceptance Criteria

- [ ] `Tests/Notes/test_notes_library_unit.py::TestNotesInteropService::test_get_db_new_instance` and `::TestLibraryNotesInteropDelegates::test_get_db_new_instance` pass, with the double updated to the real constructor signature rather than the assertion relaxed
- [ ] Both `Tests/UI/test_library_notes_lasting_sync_flow.py::test_activation_result_routes_to_truthful_receipt_or_root_recovery` parameterisations pass, with the `activate_root` double matching the production signature
- [ ] Each repaired double is verified to still fail when the behaviour it guards is broken -- a double updated only far enough to stop raising is not a fix
- [ ] A brief note records whether either double had drifted far enough that it was passing vacuously before the signature change

## Evidence (verified first-hand on dev 022b67fc7, 2026-08-23)

```
pytest Tests/Notes/test_notes_library_unit.py -k test_get_db_new_instance
  -> 2 failed, 62 deselected
pytest Tests/UI/test_library_notes_lasting_sync_flow.py -k activation_result_routes
  -> 2 failed, 5 deselected  (TypeError at test_library_notes_lasting_sync_flow.py:164)
```

Mechanisms: the Notes doubles' mock `CharactersRAGDB` signature was outgrown by TASK-19900's
`console_library_migration_seed` argument; the sync-flow double's `activate_root` is missing a
now-required positional argument.

Surfaced by the TASK-21129 implementer while A/B-baselining its own reds against dev.
