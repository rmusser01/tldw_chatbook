---
id: TASK-981
title: >-
  Audit the two @work(thread=True) async workers for cross-event-loop hazards
status: To Do
assignee: []
created_date: '2026-07-27 12:00'
labels:
  - ui
  - concurrency
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Noticed while sweeping `self.call_from_thread` misuse for TASK-929. Two workers combine `@work(thread=True)` with `async def`:

- `Widgets/Media_Creation/swarmui_widget.py:354` — `@work(exclusive=True, thread=True)` on `async def generate_image`
- `Widgets/multi_item_review_window.py:377` — `@work(thread=True)` on `async def _generate_analyses_worker`

**This is not a bug, and the task is not to "fix" it.** Textual supports the combination explicitly. `Worker._run_threaded` (`textual/worker.py:284-323`) checks `inspect.iscoroutinefunction(self._work)` and, when true, routes through `run_coroutine` → `run_awaitable` → `asyncio.run(do_work())`. The decorator only rejects the opposite mistake — a non-async function *without* `thread=True`.

**What it actually means, and why it is worth auditing.** The coroutine does not run on the application's event loop. It runs on a **brand-new event loop created by `asyncio.run()` inside the worker thread**. That is a sharp edge, because anything bound to the app's loop is now being touched from a different one:

- `asyncio` primitives created on the app loop — `Lock`, `Event`, `Queue`, `Semaphore` — are bound to that loop and misbehave when awaited from another.
- Long-lived library objects created on the app loop, notably an `httpx.AsyncClient`, carry loop-bound state; reusing one inside these workers is a genuine hazard.
- Any UI touch must go through `self.app.call_from_thread(...)` because this is a real thread. TASK-929 fixed exactly that in both of these files, which is what surfaced them.
- `asyncio.run()` also *closes* its loop on completion, so anything cached on it does not survive between invocations.

Audit both workers for those four patterns. If each only awaits objects it creates itself inside the worker, the combination is fine and should be left alone with a short comment recording why. If either shares a loop-bound object with the app, that is a real defect to fix — most cheaply by making the worker a plain `def` that owns its own `asyncio.run`, or by moving it off the thread pool entirely.

Worth checking whether any other `@work(thread=True)` in the tree decorates an `async def`; these two were found incidentally, not by an exhaustive search.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Both workers are audited for awaiting app-loop-bound asyncio primitives or clients
- [ ] Any genuine cross-loop sharing is fixed; anything safe is left alone with a comment recording why
- [ ] The tree is searched for other `@work(thread=True)` on `async def` and each is judged the same way
<!-- AC:END -->
