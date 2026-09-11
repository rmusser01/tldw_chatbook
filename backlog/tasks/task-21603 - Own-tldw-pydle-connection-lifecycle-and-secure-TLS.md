---
id: TASK-21603
title: Own tldw-pydle connection lifecycle and secure TLS
status: To Do
assignee: []
created_date: '2026-08-23 23:23'
updated_date: '2026-09-10 12:00'
labels:
  - irc
  - lifecycle
  - tls
  - security
dependencies:
  - TASK-21602
references:
  - backlog/decisions/148-network-chat-ircv3-and-tldw-pydle-boundary.md
  - backlog/decisions/149-network-chat-handoff-reliability-amendments.md
documentation:
  - Docs/superpowers/plans/2026-09-10-network-chat-pto-handoff.md
  - Docs/superpowers/plans/2026-08-23-tldw-pydle-first-release.md
  - Docs/superpowers/specs/2026-08-23-network-chat-ircv3-and-tldw-pydle-design.md
priority: high
type: task
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give embedding applications deterministic authority over connection lifetime
and verified transport security so close, cancellation, failure, and TLS
posture are explicit outcomes rather than detached side effects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The client exposes reviewed typed connect, generation-idempotent wait-ready, asynchronous close, and force-abort signatures/results from idle, connecting, negotiating, ready, failed, closing, and closed states; cancelling one close waiter does not cancel shared cleanup or another waiter's outcome.
- [ ] #2 The library performs no implicit reconnect or delayed reconnect scheduling after unexpected disconnect, authentication failure, explicit close, or cancellation.
- [ ] #3 Cooperative close refuses new commands, settles pending operations, attempts bounded QUIT where appropriate, closes and awaits the writer, cancels/drains all owned work, drains completed exceptions, and finishes with zero owned tasks and timers.
- [ ] #4 Callback code that ignores cancellation but keeps yielding yields a bounded explicitly incomplete close outcome naming the retained callback while all protocol tasks/timers reach zero and public late commands/publication are rejected; no bound is claimed for code blocking the event loop.
- [ ] #5 Direct TLS enables certificate-chain and hostname verification by default, supplies SNI, accepts a typed custom trust context, and never downgrades a failed TLS-required connection to plaintext.
- [ ] #6 Loopback certificate tests prove matching trusted success, hostname mismatch failure, untrusted-chain failure, no plaintext fallback, and bounded TLS writer closure.
- [ ] #7 Connection cancellation, read reset, failed negotiation, repeated close, stalled close, and normal close have deterministic tests with no unobserved exception or falsely clean shutdown result.
<!-- AC:END -->

## Handoff

Read the [PTO entry point](../../Docs/superpowers/plans/2026-09-10-network-chat-pto-handoff.md) before claiming this task. Implementation lives in the separate fork; this record stays in Chatbook. ADR-148 records the approved boundary; ADR-149 contains the proposed review corrections. Add an Implementation Plan only after moving this task to In Progress. No implementation or release evidence is claimed by this documentation PR.
