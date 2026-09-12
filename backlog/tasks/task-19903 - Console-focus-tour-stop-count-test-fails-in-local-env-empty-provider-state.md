---
id: TASK-19903
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

## Renumbering provenance

This task previously held id TASK-2157, colliding with the older
"Console: dictionary send agent-branch test fails (KeyError: agent_messages)"
task, which was created at 16:51 on 2026-08-07 and archived at 18:08; this task
was created one minute later, at 18:09, and the CLI reissued the id because
upstream Backlog.md hands an archived task's id straight back to the next
`task create`. Per the owner rule decided 2026-08-21 in TASK-19601 (**older id
keeps it; the younger task renumbers with a provenance note, regardless of Done
status**), it renumbered to TASK-19903. The citation in TASK-2154's
implementation notes was updated; any other reference to TASK-2157 written
before 2026-08-22 that concerns the focus-tour stop count refers to THIS task.
