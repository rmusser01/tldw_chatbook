---
id: TASK-16450
title: Shielded-cancellation loop in _execute_library_rag_search never uncancels
status: Done
assignee: []
created_date: '2026-08-15'
labels: [library, rag, async, reliability]
dependencies: []
priority: low
---

## Description (the why)

PR #1640 (TASK-15810) fixed the Library RAG first-query spin. Its worker
drains an admitted retrieval under `asyncio.shield` and retains the first
`CancelledError` to re-raise after the outcome settles
(`UI/Screens/library_screen.py`, `_execute_library_rag_search`):

```python
while not retrieval_task.done():
    try:
        await asyncio.shield(retrieval_task)
    except asyncio.CancelledError as error:
        cancellation = cancellation or error
```

Qodo (PR #1640, finding 5) observed the loop never clears the accumulated
cancellation request on the CURRENT task. A single `cancel()` is fine: one
delivery, one catch, and the next `await` blocks normally. The risk is a
caller that cancels REPEATEDLY until the task reports done — each new
`await` then raises immediately and the loop hot-spins until the retrieval
settles, delaying teardown at high CPU. Nobody has constructed that spin:
Textual's `Worker.cancel()` and asyncio's shutdown each cancel once, which
is why this was flagged on the PR rather than patched (cancellation
semantics there are load-bearing: cancelled workers must still drain the
admitted retrieval under the app-lifetime lock, then re-raise before stale
outcomes can apply). There is an irony worth respecting: this is a CPU-spin
fix carrying a potential CPU-spin under teardown.

## Acceptance Criteria (the what)

- [x] A test demonstrates the repeated-cancel scenario against the drain
      loop (or the investigation documents concretely why no caller in the
      app can produce it — with the callers named, not assumed)
- [x] If reachable: the loop clears the pending cancellation between
      catches (the 3.11+ idiom is `asyncio.current_task().uncancel()` after
      each catch — NOT `retrieval_task.uncancel()`) while preserving the
      retain-and-re-raise contract the worker's docstring states
- [x] Cancelled workers still drain the admitted retrieval before
      re-raising, and no stale outcome is applied (the existing
      Tests/UI/test_library_shell.py supersession tests stay green —
      run the targeted selection, never the whole file)

## Outcome (2026-08-18): the path is reachable; the SPIN is not

**AC#1, both halves answered — and the first correction is to this task's own
premise.** It records that nobody could construct the repeated-cancel path
because "Textual's `Worker.cancel()` and asyncio's shutdown each cancel once".
**That is wrong.** Named, from textual 8.2.8:

- `WorkerManager.add_worker(exclusive=True)` calls `cancel_group(node, group)`.
- `cancel_group` selects workers by **group and node only — no state filter**,
  and a worker that is still DRAINING is still in `_workers`.
- `Worker.cancel()` calls `task.cancel()` **unconditionally**, with no
  already-cancelled guard.

So every new exclusive Library search re-cancels a worker that is still
draining. The Library search box starts one exclusive worker per search, so an
impatient user reaches this on the ordinary path.

**But the loop does not hot-spin, and that is measured rather than argued.**
`task.cancel()` schedules ONE `CancelledError` at the next await point; the
loop catches it, loops, and the next await blocks normally until another
cancel arrives. Fifty cancellations cost fifty extra iterations and the loop
keeps making progress — bounded by user actions, not CPU-bound.
`Tests/Library/test_library_rag_drain_loop.py` constructs the path and pins
that bound (`spins <= 60`), alongside the fixed loop and the single-cancel
case.

**AC#2 applied as hygiene, not as a bug fix.** `current_task().uncancel()`
after each catch, because the path IS reachable and leaving `cancelling()`
elevated on a task that then completes normally is what an enclosing cancel
scope reads. The tests show single-cancel behaviour is unchanged and repeated
cancellation stays bounded, so the change is provably harmless rather than
assumed so — which is the bar this programme applies to cancellation
semantics.

**AC#3 preserved:** the admitted retrieval still drains and the retained
`CancelledError` is still re-raised — pinned in both the fixed and
single-cancel tests. `Tests/Library/` 2019 passed / 0 failed.

The irony the task noted survives intact and is worth keeping: a CPU-spin fix
carried a *potential* CPU-spin under teardown, and the honest answer is that
the potential was real in shape and bounded in effect.
