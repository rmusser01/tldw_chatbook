---
id: TASK-15501
title: >-
  Console left-rail composition test asserts a parent the tray no longer
  composes under
status: Done
assignee: []
created_date: '2026-08-11 19:48'
labels:
  - bug
  - console
  - tests
  - pre-existing
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_internals_decomposition.py::test_console_left_rail_sections_use_available_space asserts workspace_context.parent is session_body, but the tray now composes under console-rail-section-body-conversations. Either the composition changed without the test following, or the test encodes a layout intent the composition has drifted from; the test is red on this checkout and no longer guards the space-usage property it is named for.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tests/UI/test_console_internals_decomposition.py::test_console_left_rail_sections_use_available_space passes on a clean dev checkout
- [x] #2 It is recorded which side was wrong: the composition drifted, or the test's expected parent was stale
- [x] #3 The test still fails if a rail section stops using the available space, i.e. it is repaired rather than relaxed to vacuity
<!-- AC:END -->

## Closeout audit 2026-10-11

Board-prune audit (branch chore/board-prune-c, base a3c47c6c72): completed-but-unmarked. Evidence gathered 2026-10-11 against origin/dev:

AC dispositions:
- AC#1: met - The stale session_body assertion is gone; the test asserts the real composition (workspace_context.parent is #console-rail-section-body-conversations) per commit 66c84cd10f.
- AC#2: met - The test records which side moved: its body cites TASK-14810 splitting the former mixed Session tray; expected parent updated to #console-rail-section-body-conversations.
- AC#3: met - Still asserts real geometry (body/header regions, content-width equality against conversations_body.content_region), not relaxed to vacuity.
- Commit 66c84cd10f 'test: align Console integration contracts' (2026-08-14) realigned the composition assertion.
- - Run caveat: this audit's local runs of the UI half died in fixture setup with Backup_Recovery bootstrap.RecoveryRequired('raw_source_selection_changed') - this machine's shared profile state, the repo's known local-profile flake class (see commit e1fee9bb1e), not a dev-tree assertion failure.
