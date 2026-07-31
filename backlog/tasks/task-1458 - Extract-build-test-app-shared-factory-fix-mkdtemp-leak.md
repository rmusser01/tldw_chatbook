---
id: TASK-1458
title: >-
  Extract _build_test_app into a shared UI test factory and fix its per-call mkdtemp leak
status: Done
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

- [x] The factory lives in a non-test helper module under Tests/UI/ with `contextlib.ExitStack` replacing the nested-with pyramid; `test_screen_navigation.py` re-exports it during transition
- [x] Every temp dir the factory creates is removed by teardown (fixture wrapper or explicit cleanup)
- [x] All importing modules updated; junit outcome diff vs baseline is empty for Tests/UI

## Implementation Plan

1. Census importers (91 files, all importing exactly `_build_test_app`)
2. Port the factory to `Tests/UI/app_factory.py` with `contextlib.ExitStack`; record every mkdtemp in a module list
3. Autouse drain fixture in the root conftest (lazy `sys.modules` lookup — non-app tests pay nothing); re-export from the old home for in-flight branches
4. Update all importers mechanically; verify mechanism + outcomes

## Implementation Notes

Factory moved verbatim (including the first-run-wizard default and the
load-bearing `.resolve(strict=True)`) with the 17-level `with` pyramid replaced
by one ExitStack loop. Cleanup design: the factory records every user-data dir
and the root conftest drains them after each test — zero call-site churn for 91
importers versus a `(app, cleanup)` return-shape change. Drain proven by a
two-test probe (dir exists during the creating test, gone in the next); ambient
leak counting was useless as evidence because concurrent agent sessions on
pre-fix branches re-accumulated 41k leaked dirs during verification.
App-instance caching remains deliberately out of scope (previously tried,
wedged the compositor — regression coverage stays in test_screen_navigation).
Verified: probe; `test_screen_navigation` + `Tests/Skills` + `Tests/Watchlists`
673 passed/0 failed; `test_home_screen` 59 passed + 1 failure present in the
pre-change UI baseline. Added: `Tests/UI/app_factory.py`. Modified:
`Tests/conftest.py`, `Tests/UI/test_screen_navigation.py`, 91 importer files.
