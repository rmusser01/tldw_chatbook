---
id: TASK-707
title: 'Implement word bench concurrency, or hide the control in PR 3'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-26 14:30'
updated_date: '2026-07-27 05:52'
labels:
  - evals
  - word-bench
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the whole-branch review of PR 2 of the Evals rebuild (the word bench engine). Not a defect introduced by that PR unless stated; each is a seam the engine leaves for the screen that consumes it.

`BenchConfig.concurrency` is defined, validated into storage, and reloaded — and the runner never reads it. Execution is always sequential.

The spec makes it a requirement rather than an option: "Parallelism is opt-in through a `concurrency` field on the bench, so the setting travels with the bench that was tuned for it." Once PR 3 renders the bench editor, a user gets a control that does nothing.

This is a gap in PR 2's plan, not an implementer error — the plan defined the field in its first task and never scheduled the work. `BenchConfig.__post_init__` also validates `top_k >= 1` but not `concurrency >= 1`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Either the runner honours `concurrency` (an `asyncio.Semaphore` over the row, preserving row-major completion order), or the field is removed and PR 3 renders no control
- [x] #2 If implemented, `concurrency >= 1` is validated in `BenchConfig.__post_init__`
- [x] #3 If implemented, a test asserts a concurrency of 1 still produces strict row-major order
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement bounded parallel execution in WordBenchRunner: bound with an asyncio.Semaphore scoped to one row (all of that row's targets) at a time, preserving row-major fill and cell save order via asyncio.gather's input-order guarantee.
2. Special-case concurrency==1 as the original sequential loop, unchanged, so the tested row-major/cancel-granularity guarantees hold exactly.
3. Add concurrency >= 1 validation to BenchConfig.__post_init__.
4. Add asyncio.CancelledError handling around the run body so a hard (non-cooperative) cancellation also marks run rows cancelled and re-raises.
5. Write tests proving real parallel overlap (bounded), row-major save order under concurrency, cooperative + hard cancellation safety, and the validation error; revert-check each against the pre-fix code.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented bounded parallel execution rather than removing the field, per the spec's explicit requirement.

**Approach**: `WordBenchRunner.run` special-cases `concurrency == 1` as the original, byte-for-byte sequential double loop (no `asyncio.gather` involved at all), so every existing ordering/cancellation test keeps its exact guarantees untouched. For `concurrency > 1`, one ROW (all of that row's targets) is fanned out via `asyncio.gather`, each capture bounded by an `asyncio.Semaphore(config.concurrency)`; rows themselves are never overlapped -- row N+1 is never dispatched until row N's gather has fully returned. `asyncio.gather` returns results in submission (target-list) order regardless of completion order, so cells are always saved/progressed in `targets` order even when the underlying network calls finish out of order -- the row-major, comparable-rows contract holds at any concurrency.

**Cancellation**: the cooperative `CancelToken` is checked once per row under concurrency > 1 (not once per cell, unlike the concurrency == 1 fast path) -- a row already dispatched is allowed to finish so the grid never persists a half-captured row; documented in the module docstring as the one observable behavior change, gated behind `concurrency > 1` so no existing test is affected. Separately, a genuine `asyncio.CancelledError` (the Task running `.run()` itself hard-cancelled, e.g. by a superseding Textual worker) is now also caught directly inside the runner, marks every run row `cancelled`, and re-raises -- previously this only worked via a best-effort sweep in the ONE production caller (`sample_bench.py`'s `_mark_orphaned_runs_cancelled`); the runner now closes this gap itself too, verified by a dedicated revert-checked test using real external task cancellation (`asyncio.ensure_future(...).cancel()`), at both concurrency=1 and 2.

**Validation**: `BenchConfig.__post_init__` now rejects `concurrency < 1`, mirroring the existing `top_k` check.

**Revert-check performed**: reverted `runner.py`/`models.py` to the pre-fix `HEAD` versions and re-ran `Tests/Evals/word_bench/test_runner.py` -- exactly 4 of the 19 tests failed (the parallel-bound overlap test, the concurrency validation test, and both external-cancellation tests at concurrency=1 and 2); all others passed unaffected. Restored the fix and confirmed all 19 pass again. This confirms the new tests exercise genuinely new behavior, not vacuous assertions.

**Files**: `tldw_chatbook/Evals/word_bench/runner.py`, `tldw_chatbook/Evals/word_bench/models.py`, `Tests/Evals/word_bench/test_runner.py`.
<!-- SECTION:NOTES:END -->
