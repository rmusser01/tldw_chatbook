---
id: TASK-37100
title: >-
  Console edge-rail geometry tests error at setup under the plain command, and
  the live-resize test is red on dev
status: To Do
assignee: []
created_date: '2026-10-27 17:20'
labels:
  - console
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_edge_rail_geometry.py pins the Console's edge-owned rails: each boundary's single divider cell, the focus cue on each rail and the transcript title, and rail intent kept through a live resize. It has no bootstrap_profile marker and is not in scripts/ui_pr_gate_census.txt, so under the plain command every one of its 16 tests fails at setup on a developer machine with RecoveryRequired('raw_source_selection_changed'). PR checks do not run it; only the nightly whole-tree run does. On a developer machine those pins verify nothing today. With the marker added (a temporary copy), 15 of 16 pass and test_live_resize_reasserts_edge_ownership_and_preserves_rail_intent fails at its hit-test assertion in _assert_edge_owned_workbench, identically on origin/dev a2b7104d1e and on the TASK-33620.15.1.4 branch, so the red is pre-existing. Found while verifying TASK-33620.15.1.4 (2026-10-27).

Two more tests in the copy flaked once each at host load 30-36 and passed when rerun alone: test_native_widget_blur_repaints_the_actual_focus_successor (focus went to the setup modal's action button instead of the transcript) and test_focus_paint_clears_when_focus_becomes_none for the transcript. Once the file runs in PR checks, those two may need their waits tied to the setup card settling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test in Tests/UI/test_console_edge_rail_geometry.py reaches a pass or fail result under the plain pytest command, not a setup error
- [ ] #2 test_live_resize_reasserts_edge_ownership_and_preserves_rail_intent passes, or its expectation is corrected with the measured reason recorded in the task
- [ ] #3 The file runs in PR checks (the UI gate census), or the task records why it cannot
<!-- AC:END -->
