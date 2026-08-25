---
id: TASK-22205
title: >-
  Move the per-send durable-turn commit and restore reconcile off the event loop
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - console
  - database
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22205).

New since the pin (console dispatch checkpoint work). Per send,
`Chat/console_chat_controller.py:5635` runs `store.commit_durable_turn(acceptance)`
synchronously on the event loop before the provider request goes out: one
`BEGIN IMMEDIATE` transaction (`Chat/chat_persistence_service.py:384`) containing ~10
statements including two message INSERTs (each firing the FTS trigger and a full-content
`sync_log` JSON write — ~3x write amplification of the message text), an attachments
`executemany`, and a readback (`Chat/console_dispatch_repository.py:103-262`). A second
IMMEDIATE transaction runs at `:657` pre-dispatch and a third at settle
(`console_chat_store.py:2141`). Separately, every conversation restore/switch runs
`reconcile_for_session` (`console_dispatch_repository.py:324-345`) — an IMMEDIATE (write)
transaction taken to read recovery state that usually writes nothing, plus a recursive
active-path CTE, inline on the loop. Steady-state cost is tens of ms; under the 22200
backfill window it is unbounded up to the 15 s busy timeout — on the event loop.

## Acceptance Criteria

- [ ] The durable-turn commit runs off the event loop (worker/`to_thread`) with its ordering guarantees preserved and the dispatch-checkpoint test suite green
- [ ] `reconcile_for_session` takes a write transaction only when it actually has a write to make; the read path uses a read transaction
- [ ] Send-to-dispatch latency measured before/after (steady state and with an artificial write-lock holder)
- [ ] No new shutdown/error-path regressions: the review's standing rule — when you defer, walk the teardown and failure paths in real teardown order
