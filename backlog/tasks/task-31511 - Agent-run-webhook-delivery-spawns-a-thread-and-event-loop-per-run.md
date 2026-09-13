---
id: TASK-31511
title: Agent-run webhook delivery spawns a thread and event loop per run
status: To Do
assignee: []
created_date: '2026-09-04 19:30'
labels:
  - performance
  - agents
dependencies: []
priority: low
---

## Description (the why)

`Agents/run_webhooks.py` (`schedule_run_webhook`) spawns a fresh
`threading.Thread` whose target calls `asyncio.run(deliver_webhook(...))` --
a new OS thread plus a new event loop per delivery -- and
`_maybe_emit_run_webhook` (`agent_service.py:3264`) re-reads settings from
disk on every run completion. Opt-in and low-frequency, so low priority; the
shape is the issue (unbounded thread creation under bursty fleets, settings
re-read per event). Evidence:
`Docs/Design/2026-09-04-holistic-perf-review.md` section 7.

## Acceptance Criteria (the what)

- [ ] Webhook deliveries reuse an executor or long-lived worker rather than a fresh thread + event loop per run
- [ ] Settings are not re-read from disk per completion when unchanged (cache with invalidation, or read once per run batch)
- [ ] Delivery semantics (fire-and-forget, never blocks run finalization) are unchanged
