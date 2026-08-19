---
id: TASK-18939
title: 'Scheduling: per-task execution timeout knob'
status: To Do
assignee: []
created_date: '2026-08-19 11:05'
updated_date: '2026-08-19 11:05'
labels:
  - scheduling
  - parity
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the timeout-configurability gap found by the TASK-18936 parity audit (hermes's cron hardening made timeout configurable and derived run-claim TTL from it; chatbook's only scheduling knob is `[scheduling] scheduler_poll_interval_seconds`). Add a per-task (with a global default) execution timeout: a dispatched task whose handler exceeds the bound is marked failed-with-timeout rather than wedging the loop or hanging indefinitely. Note the current dispatch reality before designing: `SchedulerLoop.tick` awaits handlers sequentially, and `BriefingJobHandler` is deliberately fire-and-forget — so the timeout must first characterize where long-running work actually blocks (a wedged `watchlist_job`, a future synchronous agent-task handler) and apply the bound at the right seam, not blindly wrap every handler in `asyncio.wait_for`. Manual "Run now" (TASK-18938) and missed-fire accounting (TASK-18937) should consume the same timeout outcome.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A global default timeout plus per-task override exist in config (naming follows existing `[scheduling]` conventions) with a documented default and 0/negative semantics pinned (disable vs unlimited — pick and test one)
- [ ] #2 A characterization of which handlers can actually exceed a timeout is recorded in the task (which are fire-and-forget, which block the tick) before the bound is applied
- [ ] #3 Timed-out dispatches produce a distinct, honest terminal state (`last_status` value naming timeout) and never wedge the poll loop; the next occurrence still computes correctly afterward
- [ ] #4 Run-now and missed-fire accounting treat a timeout as a failed dispatch, not a missed one (ran-and-raised vs never-ran distinction shared with TASK-18937)
- [ ] #5 Tests use the real loop/handler shapes (a deliberately slow handler in tests) and pin loop liveness after a timeout
- [ ] #6 Docs: schedules.md and the config reference document the knob and default
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: bounded-execution guard within the existing module; ADR-031-style conventions and ADR-018's module boundary are untouched.

1. Characterize handler blocking behavior (tick awaits; briefing is fire-and-forget; watchlist/reminder shapes)
2. Config knob + timeout enforcement at the seam the characterization justifies
3. Terminal state + interplay with Run-now/missed-fire tasks
4. Slow-handler tests + docs
<!-- SECTION:PLAN:END -->
