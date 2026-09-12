---
id: TASK-32310
title: >-
  Library test flakes: order-dependent Collections rail-count test and the Notes
  loader recompose race under load
status: To Do
assignee: []
created_date: '2026-09-11 00:56'
labels:
  - library
  - test-health
  - flaky
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two reds recur across the critique-9 wave PRs and are caused by none of them. (1) Tests/UI/test_library_crit8_collections_row.py::test_rail_count_never_falls_back_to_the_unfiltered_total_after_a_visit is ORDER-DEPENDENT: it passes only on state left by a test that runs before it and fails identically on plain origin/dev in a clean worktree (proved on PR #2581's bot round). (2) Tests/UI/test_library_crit8_notes_loader.py and the row-press test in test_library_canvas_sync_defects.py intermittently hit WorkerFailed NoMatches('#library-note-title' on LibraryNoteWorkPane) under host load, the recompose race the canvas's own _after_recompose guard documents; a 5x5 A/B on PR #2571 showed equal rates on the branch and on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The rail-count test passes alone and in any order, or sets up its own state
- [ ] #2 The loader recompose race is closed at the source, or the affected tests wait on the real seam rather than a timing window
<!-- AC:END -->
