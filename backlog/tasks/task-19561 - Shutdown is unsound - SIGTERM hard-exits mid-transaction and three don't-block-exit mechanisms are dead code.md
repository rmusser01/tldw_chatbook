---
id: TASK-19561
title: >-
  Shutdown is unsound — SIGTERM hard-exits mid-transaction and three
  "don't block exit" mechanisms are dead code
status: To Do
assignee: []
created_date: '2026-08-21 20:11'
labels:
  - concurrency
  - shutdown
  - data-integrity
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 4 (concurrency / async / workers) —
its **#3**, **#6** and **#8**. Grouped: one fix locus, the app shutdown path.
Re-verified at this branch base.

**A — SIGTERM skips the entire shutdown.** CONFIRMED.
`app.py:13083` calls `os._exit(0)` from the handler installed at
`app.py:13086` (`signal.signal(signal.SIGTERM, signal_handler)`). `os._exit`
bypasses every cleanup path. Consequences the lane traced: an in-flight
watchlist transaction is abandoned **mid-write**, and a run row is left stuck
at `'running'` **forever** — there is no startup sweep that reconciles it.
This is the *live* path, not a corner case: **SIGINT is consumed by Textual's
raw mode**, so SIGTERM is how this app actually gets terminated.

**B — three "don't block exit" mechanisms are all dead code.**
CONFIRMED BY PROBE — `app.py:11901-11910`, `13040-13051`, `13054-13059`. Each
tries to set `thread.daemon = True` on threads that are **already started**,
which Python forbids: the lane's probe **raised `RuntimeError` and the daemon
flag was unchanged**. One of the three logs an ERROR per thread on **every**
exit, so the mechanism is not merely inert, it is noisy.
Measured consequence: interpreter exit blocked for **3.00 s** after
`asyncio.run` returned at 0.30 s. Compounding it, `app.py:11879` sets
`loop._default_executor = None`, which defeats the graceful executor shutdown
that would otherwise help here.

**C — scheduled briefing generations are invisible to shutdown.** CONFIRMED.
`Scheduling/scheduler/handlers/briefing_handler.py:162` spawns via
`asyncio.create_task(self._run_generation(watchlist_id))`. These are not
Textual workers, so they are absent from the only collection shutdown cancels.
The result is a wedged `'generating'` row whose recovery is UI-gated.

Also worth folding in while here (lane #12, lower severity): a flat 100 ms
shutdown wait, and bare `create_task` calls with no retained reference at
`app.py:10010` and `Voice_Cloning_Window.py:701-719` — a task with no strong
reference can be garbage-collected mid-flight.

Per the owner's standing ruling, prefer the durable fix — make shutdown
actually wait for and cancel what it owns — over widening the hard-exit or
tuning the 100 ms sleep.

## Acceptance Criteria

- [ ] SIGTERM runs the ordinary shutdown path: in-flight database transactions
      are completed or rolled back, not abandoned mid-write
- [ ] A hard exit remains available as a **last-resort escape after** the
      graceful path has been given a bounded chance, not as the first action
- [ ] No run/generation row can be left stuck in `'running'`/`'generating'` by
      a termination; either shutdown reconciles it, or a startup sweep does
      (a startup sweep is the durable option — it also covers power loss)
- [ ] The three dead `thread.daemon = True` mechanisms are removed or replaced
      with something that works; nothing logs an ERROR per thread on a normal
      exit
- [ ] Interpreter exit is not blocked for seconds after the event loop stops —
      measured, with the before/after numbers recorded
- [ ] `loop._default_executor = None` is re-examined against the graceful
      executor shutdown it defeats
- [ ] Scheduled briefing generations are tracked somewhere shutdown can see and
      cancel them
- [ ] Bare `create_task` results that must outlive their caller retain a strong
      reference
