---
id: TASK-31511
title: Agent-run webhook delivery spawns a thread and event loop per run
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 19:30'
updated_date: '2026-09-12 18:07'
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
- [x] #1 Admission is nonblocking and bounded to one delivery plus 32 queued events; overflow refuses the newest event without logging payloads or secrets.
- [x] #2 One worker reuses its event loop, retires after 30 idle seconds, and handles admission/retirement races and failed starts without stranding accepted events.
- [x] #3 Each accepted event retains its event-time configuration and identifiers; existing signature, SSRF, subscription and timeout behavior remains covered.
- [x] #4 Webhook deliveries reuse a bounded worker and event loop rather than a fresh thread and event loop per run.
- [x] #5 Unchanged settings are not re-read from disk per completion; the existing cache invalidation remains correct.
- [x] #6 Delivery remains best effort and never blocks run finalization; admission is distinct from successful delivery.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/153-bounded-run-webhook-delivery.md. Reason: bounded admission, worker lifetime, and drop semantics are runtime ownership decisions. Execute Docs/superpowers/plans/2026-09-12-bounded-run-webhooks.md: gated baseline/TDD, reusable bounded worker, lifecycle/security regression tests, existing config-cache verification, documentation and independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-153 with one lazy module-owned webhook worker: 32 waiting notifications plus one delivery, FIFO, one asyncio Runner per generation and 30-second idle retirement. Admission returns immediately after acceptance/refusal; full or retiring queues refuse newest work. Exact-generation retirement closes the Runner before publishing retirement; failed starts and individual delivery failures remain contained. Immutable event-time configuration and copied identifiers preserve routing/signing. New diagnostics use fixed wording, bounded event/reason labels and exception classes; diagnostic sink failures cannot escape admission or terminate FIFO processing.

Existing payload/HMAC/SSRF/subscription/timeout/missing-secret behavior is unchanged. The Console agent-runs guide distinguishes admission from successful delivery and documents best-effort drops, process-exit loss and no retry/outbox. No application lifetime service, new cache, dependency, migration or shutdown join was introduced. Only the reviewed derived diagnostic inventory was regenerated; no guard limits changed.

Evidence: the affected webhook/config-cache/hot-reload selection passed 35 tests; after independent review corrected test ownership, the final webhook module passed 25 tests in 0.70s. Tests exercise shared thread/loop identity, bounded nonblocking FIFO admission, close/admission race, restart/start failure, immutable capture, callback/diagnostic failure containment and sensitive diagnostic canaries. Every successful test generation is captured as an exact Thread and joined in finally before dependencies are restored, including the isolated enabled-scheduler worker. The selections overlap; do not sum them as unique tests. Scoped Ruff, format, whitespace and diagnostic inventory checks passed. Existing RequestsDependencyWarning and foreign pytest garbage-directory cleanup warnings remain visible.

Root confirmed unchanged terminal persistence ordering: AgentService schedules after a successful AgentRunsDB transaction returns. The terminal-mapping test verifies event categories; it is not a fresh end-to-end persistence/delivery test. Existing cache/invalidation tests qualify warm settings reads; no duplicate cache is needed. Deterministic transports and isolated settings were used, with no real outbound webhook or full test sweep.

Implementation 5e7dac433a, cleanup correction ff052efe93. Independent task review and scoped re-review approved; unchanged-code verification items resolved. ADR required: yes; ADR path: backlog/decisions/153-bounded-run-webhook-delivery.md. Broader orchestration branch integration review remains part of the parent workstream.
<!-- SECTION:NOTES:END -->
