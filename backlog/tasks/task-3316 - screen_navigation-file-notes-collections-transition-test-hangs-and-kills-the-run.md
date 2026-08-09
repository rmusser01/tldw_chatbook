---
id: TASK-3316
title: >-
  screen_navigation file-notes collections transition test hangs and kills the whole run
status: To Do
assignee: []
created_date: '2026-08-08 21:30'
labels:
  - tests
  - file-notes
  - dev-baseline
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during follow-up batch phase A (2026-08-08): `Tests/UI/test_screen_navigation.py::test_file_notes_collections_source_transition_blocks_mutation_through_recompose` hangs on dev base `ebeae1440` (reproduced with the phase's product diff fully reverted). Under the repo's `timeout_method = thread` a hung test dumps stacks and terminates the ENTIRE pytest process (the task-1466 lesson), so any run that collects this file dies — which also hides every test after it. Belongs to the file-notes/collections surface, not the ingest arc; filed from the ingest batch so it does not rot unowned.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The hang's mechanism is identified and fixed (or the test bounded at its source) so the file completes under the standard timeout
- [ ] #2 Full `Tests/UI/test_screen_navigation.py` completes with a READ pass count
<!-- AC:END -->
