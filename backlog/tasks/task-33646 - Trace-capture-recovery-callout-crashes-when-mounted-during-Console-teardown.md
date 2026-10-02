---
id: TASK-33646
title: Trace-capture recovery callout crashes when mounted during Console teardown
status: To Do
assignee: []
created_date: '2026-10-02 01:03'
labels:
  - console
  - flaky-test
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TraceCallRecoveryCallout.on_mount calls sync_recovery, which query_one()s its fixed buttons (#console-trace-save-send first). Under CI load the callout can mount while its screen is being torn down, so the children are already gone: NoMatches is raised from on_mount and recorded as event=unhandled_exception component=app. Seen in Perf Guard (Tests/Performance/test_textual_css_fastpath.py::test_ancestor_scoped_bare_type_rule_count_is_a_ratchet) on #2903's runs 36946056592 and 36946671909 (2026-10-02); a re-run of the same head passed, so it is timing-dependent, not caused by that PR's change. An unhandled exception there is an app-level crash path, not only a test flake.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mounting or syncing the callout after its children are gone (screen teardown) neither raises nor logs an unhandled exception
- [ ] #2 A test drives the callout's sync after its children are removed and pins the no-raise behaviour
- [ ] #3 The CSS fast-path ratchet test no longer fails on this race
<!-- AC:END -->
