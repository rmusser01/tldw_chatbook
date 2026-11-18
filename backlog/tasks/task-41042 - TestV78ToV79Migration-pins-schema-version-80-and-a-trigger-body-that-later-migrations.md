---
id: TASK-41042
title: "TestV78ToV79Migration pins schema version 80 and a trigger body that later migrations changed"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- database
priority: low
references:
- Tests/ChaChaNotesDB/test_sargable_timestamps.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/ChaChaNotesDB/test_sargable_timestamps.py::TestV78ToV79Migration` asserts that the ChaChaNotes schema is at version 80 after migration. `_CURRENT_SCHEMA_VERSION` is 86 on dev ecad404bd4 (2026-11-15), so three of its tests fail on that number. A fourth, `test_migration_is_idempotent`, fails on a trigger body that a later migration changed.

The class still exercises the v78 to v79 step, but its pins of the final schema state went stale. TASK-36949.2's review saw the same failures on 2026-11-06. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The class checks the v78 to v79 step without pinning the current schema version or triggers that later migrations own
- [ ] #2 All four tests pass on dev
<!-- AC:END -->
