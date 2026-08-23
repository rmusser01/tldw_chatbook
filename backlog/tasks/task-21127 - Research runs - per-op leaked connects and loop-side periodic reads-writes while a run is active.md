---
id: TASK-21127
title: >-
  Research runs - per-op leaked connects and loop-side periodic reads/writes while a run is active
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - research
  - database
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21127).

`Research_Interop/local_research_service.py:99-123` opens per-op and GC-leaks connections
(~with conn: sites); the engine is launched as a loop coroutine (Research_Window.py:594,
chat_screen.py:16200 - run_worker without thread), with a 30 s lease WRITE
(local_research_engine.py:387-393) and a 2 s `get_run` read poll (Research_Window.py:816-831)
on the loop while a run is active.

## Acceptance Criteria

- [ ] The service holds a thread-local connection; engine service calls route through to_thread
- [ ] The 30 s keepalive is batched with progress writes; the 2 s auto-refresh reads off-loop
- [ ] Research behavior unchanged - existing tests green
