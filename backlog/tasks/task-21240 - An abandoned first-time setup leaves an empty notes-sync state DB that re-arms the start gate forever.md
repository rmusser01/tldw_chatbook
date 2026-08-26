---
id: TASK-21240
title: >-
  An abandoned first-time setup leaves an empty notes-sync state DB that
  re-arms the start gate forever
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - bug
  - notes-sync
  - startup
  - performance
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; the residual
recorded when TASK-21112 merged (PR #2009, `30c7e1fe9`).

TASK-21112 gated the notes-sync runtime start on real evidence of configuration — a
zero-profile boot now creates no state file at all — and backed the full-tree watcher off from
60 to **8 scans/min** when quiet. The gate predicate (`app.py`, `start_evidence`) is
`legacy_sync_directory_configured(settings) or state_path.exists()`, deliberately side-effect
free: the obvious probe would have created the store it was testing for, which is the trap
that implementation hit and recorded in the lessons.

The residual: `state_path.exists()` is satisfied by an **empty** state database. A user who
opens first-time setup — which force-starts the runtime on demand and therefore creates the
file — and then abandons it without configuring any sync directory is left with an empty store
on disk. Every subsequent boot sees the file, arms the gate, and pays the full start the gate
exists to avoid, permanently, for a feature that was never configured.

## Acceptance Criteria

- [ ] A boot following an abandoned first-time setup does not start the notes-sync runtime
- [ ] The gate predicate remains side-effect free — evaluating it never creates, opens for write, or migrates the state store
- [ ] A genuinely configured profile, and the one-time legacy `[notes]` sync-directory migration path, still start exactly as they do today
- [ ] A test covers the abandoned-setup boot and fails if the runtime starts
