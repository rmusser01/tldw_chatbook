---
id: TASK-15663
title: 'Repair the local-marks v16 to v17 migration schema test (unowned dev red)'
status: To Do
assignee: []
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
- [ ] #1 The test passes on dev, or is deleted with a recorded reason if the contract it asserts no longer exists
- [ ] #2 The diagnosis states whether the schema or the expectation was wrong, with evidence
- [ ] #3 Tests/Chat runs with zero failures attributable to this file
<!-- AC:END -->
