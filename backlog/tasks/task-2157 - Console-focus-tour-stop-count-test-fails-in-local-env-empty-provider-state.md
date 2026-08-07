---
id: TASK-2157
title: 'Console: focus-tour stop-count test fails in local env (empty-provider state)'
status: To Do
assignee: []
created_date: '2026-08-07 18:09'
labels:
  - console
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_tab_scope.py::test_console_focus_tour_reaches_transcript_chips_inspector_under_ten_stops fails in this local dev env: the tour reaches console-empty-provider-action (setup empty state) instead of the status chips -- the env boots Console into provider-not-configured state, changing the focus landscape the tour budget was tuned for (shipped in TASK-2154.11, AC-02). Verified PRE-EXISTING: fails identically at 844966c5d (pre-batches-4/5 baseline) and at 6dc8d41a8. Possibly shares an environmental root cause with TASK-2156 (first-run wizard layout in local env): the test may need to pin a configured-provider fixture instead of inheriting ambient config.
<!-- SECTION:DESCRIPTION:END -->
