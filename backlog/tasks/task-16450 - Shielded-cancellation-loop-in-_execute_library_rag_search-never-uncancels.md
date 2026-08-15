---
id: task-16450
title: Shielded-cancellation loop in _execute_library_rag_search never uncancels
status: To Do
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

- [ ] A test demonstrates the repeated-cancel scenario against the drain
      loop (or the investigation documents concretely why no caller in the
      app can produce it — with the callers named, not assumed)
- [ ] If reachable: the loop clears the pending cancellation between
      catches (the 3.11+ idiom is `asyncio.current_task().uncancel()` after
      each catch — NOT `retrieval_task.uncancel()`) while preserving the
      retain-and-re-raise contract the worker's docstring states
- [ ] Cancelled workers still drain the admitted retrieval before
      re-raising, and no stale outcome is applied (the existing
      Tests/UI/test_library_shell.py supersession tests stay green —
      run the targeted selection, never the whole file)
