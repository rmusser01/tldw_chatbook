---
id: TASK-1900
title: 'Console UI suites: one test fails per full-sweep run under random ordering'
status: To Do
assignee: []
created_date: '2026-08-02 08:00'
labels:
  - testing
  - flaky
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Running the ~60 approvals-related suites under `Tests/UI`, `Tests/Agents` and `Tests/Chat` together produces exactly **one** failure per run — and a *different* test each time. Both observed failures pass in isolation.

Observed twice, on two different commits:

| commit | failure |
|---|---|
| `22403cb47` (dev) | `test_console_internals_decomposition.py::test_console_staged_context_tray_stays_quiet_when_populated` |
| `5aa75141d` (dev, one PR earlier) | `test_console_native_chat_flow.py::test_console_conversation_browser_search_ignores_stale_results` * |

\* that second one turned out to be a genuine deterministic failure at `5aa75141d`, already fixed on current dev — so the *reproducible* instance of this task is the staged-context-tray one.

The staged-context tray test stages into `app.pending_handoffs` **before** `run_test`, so it depends on shared app-level state being clean at entry. A prior test leaving a handoff staged (or consuming one) is the obvious shape, though the obvious candidate (`test_console_live_work_handoffs.py`) did **not** reproduce it when paired directly.

Cost: a full sweep is not currently a reliable signal — a single red test that passes on re-run trains everyone to ignore it, which is exactly how the two stale contract tests in TASK-1861 sat unnoticed on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The polluting test (or shared-state seam) is identified by name, not guessed at
- [ ] #2 The sweep passes repeatedly under several different random seeds
- [ ] #3 The shared state involved is reset per test rather than the victim test being made tolerant of dirt
- [ ] #4 A regression test fails if that shared state leaks between tests again
<!-- AC:END -->

## Update: at least one test is intrinsically flaky, not order-dependent

`Tests/UI/test_console_native_chat_flow.py::test_console_conversation_browser_search_ignores_stale_results` fails roughly **1 run in 5 when run entirely alone**, same commit, same file contents, `-p no:randomly`:

```
run 1: 1 failed   run 2: 1 passed   run 3: 1 passed   run 4: 1 passed   run 5: 1 passed
```

Failure is always `Browser row 'fresh-beta' not found. Rows: [('stale-alpha', ...)]` -- the search result arrives after the assertion, so it is a missing await/settle, not pollution.

This cost real time: it failed on a feature branch, passed on that branch's base in the same environment, and the obvious conclusion ("my change broke it") was wrong. Bisecting a diff against a coin-flip test is unfalsifiable -- **run a suspect test 5x before attributing a failure to a change.**

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduce with a captured seed, not by re-running and hoping. `pytest-randomly` prints `Using --randomly-seed=N` in the header — **capture the full output**; the first investigation grepped the run down to three lines and discarded the seed, which is why this task carries no reproducer.

With a seed, `pytest -p randomly --randomly-seed=N` replays the exact order, and `pytest --lf` plus bisecting the file list narrows the polluter quickly.

AC#3 matters: making the victim tolerant (e.g. clearing `pending_handoffs` in its own setup) hides the leak for the next test to trip over.
<!-- SECTION:NOTES:END -->
