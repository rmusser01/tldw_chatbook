---
id: TASK-32489
title: Implement bounded conversation progress inboxes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 23:53'
updated_date: '2026-09-11 00:17'
labels:
  - agents
  - console
dependencies:
  - TASK-32022
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide a bounded in-memory child-progress queue with scoped capabilities and exact lifecycle cleanup under ADR-136.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Queues enforce child, conversation, runtime, lifetime, and serialized bounds atomically.
- [x] #2 Collection preserves complete messages and scope; user discard and lifecycle cleanup release pending capacity exactly once.
- [x] #3 Concurrent boundary and stale-capability regressions pass without changing existing steering behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md
Reason: Implements the accepted reviewed contract.

1. Follow Task 1 in Docs/superpowers/plans/2026-09-10-scoped-agent-messaging.md.
2. Add failing real-owner tests for bounds, identity, collection, discard, concurrency and lifecycle.
3. Implement the stdlib inbox store and optional coordinator binding/revocation under the fixed lock order.
4. Run targeted tests and scoped Ruff/format checks; review the before-image diff and record evidence. No commit or staging.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-136 session-only MessageStore with bounded conversation inboxes, exact sender/reader capabilities, complete serialized collection, selected-ID discard, and runtime accounting. Coordinator binding freezes child identity once and revokes posting atomically on finish; reports survive terminal pruning.

Validation: 65 owner/coordinator tests passed, including controlled concurrency and stale-capability cases. Required combined run: 81 passed, 6 pre-existing steering metadata assertion mismatches assigned to the runtime integration follow-up; production steering unchanged. New-file lint/format checks pass; coordinator retains its same five baseline diagnostics. Independent spec/quality review approved with no blocker. Environment dependency/pytest cleanup warnings remain outside scope.

ADR: backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md; no new ADR needed. Full evidence: .superpowers/sdd/2026-09-10-scoped-agent-messaging/task-1-report.md and task-1-review.md. Runtime tools and UI remain subsequent implementation scope. No commit or staging.
<!-- SECTION:NOTES:END -->
