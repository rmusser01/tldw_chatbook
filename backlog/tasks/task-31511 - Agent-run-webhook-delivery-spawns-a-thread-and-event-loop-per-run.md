---
id: TASK-31511
title: Agent-run webhook delivery spawns a thread and event loop per run
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 19:30'
updated_date: '2026-09-12 16:29'
labels:
  - performance
  - agents
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bound optional run-webhook delivery resources so bursts of completed fleet runs cannot create an unbounded number of threads or pending notifications. The existing process-wide settings cache already avoids unchanged warm-path disk reads; verify its invalidation behavior while preserving the existing signed, best-effort delivery contract. Original performance finding: Docs/Design/2026-09-04-holistic-perf-review.md section 7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Admission is nonblocking and bounded to one delivery plus 32 queued events; overflow refuses the newest event without logging payloads or secrets.
- [ ] #2 One worker reuses its event loop, retires after 30 idle seconds, and handles admission/retirement races and failed starts without stranding accepted events.
- [ ] #3 Each accepted event retains its event-time configuration and identifiers; existing signature, SSRF, subscription and timeout behavior remains covered.
- [ ] #4 Webhook deliveries reuse a bounded worker and event loop rather than a fresh thread and event loop per run.
- [ ] #5 Unchanged settings are not re-read from disk per completion; the existing cache invalidation remains correct.
- [ ] #6 Delivery remains best effort and never blocks run finalization; admission is distinct from successful delivery.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/153-bounded-run-webhook-delivery.md. Reason: bounded admission, worker lifetime, and drop semantics are runtime ownership decisions. Execute Docs/superpowers/plans/2026-09-12-bounded-run-webhooks.md: gated baseline/TDD, reusable bounded worker, lifecycle/security regression tests, existing config-cache verification, documentation and independent review.
<!-- SECTION:PLAN:END -->
