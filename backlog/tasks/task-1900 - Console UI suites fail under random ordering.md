---
id: TASK-1900
title: 'Console UI suites: one test fails per full-sweep run under random ordering'
status: In Progress
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
- [x] #1 REWRITTEN -- the premise was wrong. It is not pollution: the test fails ALONE. Cause characterised below.
- [ ] #2 `test_console_conversation_browser_search_ignores_stale_results` passes 20 consecutive runs alone
- [ ] #3 It also passes with four CPU burners alongside, which reproduces it 3/3 today
- [ ] #4 The root cause is named -- why the rendered row list does not converge to a state that is provably already correct
- [ ] #5 Whether this is reachable by a real user (a search whose results never repaint) is answered yes or no, with evidence
<!-- AC:END -->

## Update: at least one test is intrinsically flaky, not order-dependent

`Tests/UI/test_console_native_chat_flow.py::test_console_conversation_browser_search_ignores_stale_results` fails roughly **1 run in 5 when run entirely alone**, same commit, same file contents, `-p no:randomly`:

```
run 1: 1 failed   run 2: 1 passed   run 3: 1 passed   run 4: 1 passed   run 5: 1 passed
```

Failure is always `Browser row 'fresh-beta' not found. Rows: [('stale-alpha', ...)]` -- the search result arrives after the assertion, so it is a missing await/settle, not pollution.

This cost real time: it failed on a feature branch, passed on that branch's base in the same environment, and the obvious conclusion ("my change broke it") was wrong. Bisecting a diff against a coin-flip test is unfalsifiable -- **run a suspect test 5x before attributing a failure to a change.**

## Investigation (2026-08-02): three hypotheses killed

**Not pollution.** Fails run entirely alone, `-p no:randomly`, ~1 in 5.

**Not a product race in the search guard.** `_refresh_console_conversation_browser_search` re-checks both the token and the query AFTER its await, before mutating. Verified by instrumenting the failing run: at the moment of failure `_console_conversation_browser_rows` holds exactly `['fresh-beta']` -- the correct, fresh result. The stale result does NOT win.

**Not the wait budget.** The helper spun a fixed `for _ in range(80)` over `pilot.pause(0.05)` -- "four seconds" that shrinks precisely when the machine is busy. Replacing it with a 15-SECOND WALL-CLOCK deadline did not help: still fails, and the rendered list still shows only `stale-alpha` after the full 15s. An explicit `_sync_console_workspace_context()` + pause before the wait fixed it only 1 run in 3.

**What is left, and it is the interesting part:** the screen's row STATE is correct and the RENDERED row list never converges to it, given 15 seconds. That is either a missed refresh/recompose trigger on the browser row list, or something in the harness that stops pumping -- not yet distinguished, which is AC#4. AC#5 matters because if it is the former, a real user's conversation search can show stale rows indefinitely.

Reproducer: run the test with four busy-loop processes alongside -- 3/3 failures. Alone, ~1 in 5.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduce with a captured seed, not by re-running and hoping. `pytest-randomly` prints `Using --randomly-seed=N` in the header — **capture the full output**; the first investigation grepped the run down to three lines and discarded the seed, which is why this task carries no reproducer.

With a seed, `pytest -p randomly --randomly-seed=N` replays the exact order, and `pytest --lf` plus bisecting the file list narrows the polluter quickly.

AC#3 matters: making the victim tolerant (e.g. clearing `pending_handoffs` in its own setup) hides the leak for the next test to trip over.
<!-- SECTION:NOTES:END -->
