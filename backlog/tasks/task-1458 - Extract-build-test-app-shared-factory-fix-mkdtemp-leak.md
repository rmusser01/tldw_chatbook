---
id: TASK-1458
title: >-
  Extract _build_test_app into a shared UI test factory and fix its per-call mkdtemp leak
status: To Do
assignee: []
created_date: '2026-07-30 08:55'
labels:
  - testing
  - ui
priority: high
dependencies: [task-1457]
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`_build_test_app()` lives inside `Tests/UI/test_screen_navigation.py:785` yet is imported by 60+ test modules and called 1,344 times per full run. It builds the real TldwCli under ~15 nested `unittest.mock.patch` context managers and calls `tempfile.mkdtemp` with no cleanup — 1,344 orphaned temp dirs per run. A test module is the wrong home for shared infrastructure. Note: session-scoped app REUSE is explicitly out of scope — it was tried before and reverted (see the wedged-compositor regression test); this task is extraction + hygiene only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] The factory lives in a non-test helper module under Tests/UI/ with `contextlib.ExitStack` replacing the nested-with pyramid; `test_screen_navigation.py` re-exports it during transition
- [ ] Every temp dir the factory creates is removed by teardown (fixture wrapper or explicit cleanup)
- [ ] All importing modules updated; junit outcome diff vs baseline is empty for Tests/UI
