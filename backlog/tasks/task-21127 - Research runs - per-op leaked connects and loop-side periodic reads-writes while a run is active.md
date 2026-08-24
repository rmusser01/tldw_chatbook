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

## Re-verification against dev 2be18842a (2026-08-23)

An independent read-only pass re-checked this finding. **All three legs still true; line cites
have drifted; one prescribed fix has a data-loss shape.**

**Confirmed, with corrected cites** (the filed `99-123` is now the schema-deferral block):
- `Research_Interop/local_research_service.py:124-158` — fresh connection per operation in
  file-backed mode, re-running `PRAGMA journal_mode = WAL` and `synchronous = NORMAL` on every
  open. **21** `with self._connect() as conn:` sites, **one** `.close()` in the file; `with conn:`
  is a transaction manager, not a closer, so the rest are GC-leaked.
- **Worse than filed**: `_update_row` (`:429-450`) opens **three** connections per single update
  (`_require_one`, the UPDATE, then `_require_one` again), each paying the private seam's
  owner-policy validation and `verify_trusted_directory`.
- `UI/Research_Window.py:595-600` — `run_worker(_run_engine(), ...)` with no `thread=True`.
- Every DB method on `LocalResearchService` is synchronous and the engine calls ~40 of them
  directly, so unlike an earlier finding in this programme, **an offload here would move real
  work** rather than zero.
- Keepalive: `local_research_engine.py:371-400` — a synchronous WRITE on the loop every 30 s for
  the life of a run. 2 s poll confirmed at `Research_Window.py:816`, correctly gated to an active
  non-terminal local run, reaching a `_call_service` seam that evaluates the method synchronously
  before its `await` plus an uncached `inspect.signature()` per call.

**Trap for the implementer**: do NOT naively add `conn.close()` to the 21 sites. In `:memory:`
mode `_open_connection` (`:137-148`) returns the **shared** `self._memory_conn`, and closing it
destroys the database; `close()` (`:161-165`) is the only legitimate closer.

**Revised severity**: every leg is real, but the whole surface is gated behind "user opens the
Research screen and starts a local run" — not a boot, keystroke or per-frame cost.
