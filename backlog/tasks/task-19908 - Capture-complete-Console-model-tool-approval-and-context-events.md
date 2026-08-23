---
id: TASK-19908
title: Capture complete Console model tool approval and context events
status: Done
assignee: []
created_date: '2026-08-22 18:28'
updated_date: '2026-08-22 23:52'
labels: []
dependencies:
  - TASK-19907
references:
  - >-
    Docs/superpowers/specs/2026-08-22-task-19907-trace-v2-exhaustive-collaboration-design.md
  - >-
    Docs/superpowers/plans/2026-08-22-task-19907-19910-trace-v2-event-foundation.md
  - >-
    backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Instrument the Console runtime so every observable user, system/context, model, streaming, tool, approval, retrieval, compaction, retry, cancellation, and failure transition is durably recorded by its existing owner and normalized by the Trace v2 projection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every documented Console event family emits start, outcome, and failure or cancellation records where applicable
- [x] #2 Tool approval decisions, provider retries/errors, RAG/context injection, and compaction ancestry are captured with causal links
- [x] #3 Sensitive payload fields are classified at capture time and hidden reasoning content is never persisted
- [x] #4 Trace capture failures never fail the user run and are surfaced through diagnostics
- [x] #5 Real-seam integration tests prove ordered capture across a representative Console run
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md. Reason: this task directly implements the existing exhaustive capture/privacy contract without changing storage ownership or cross-module architecture. 1. Add failing table-driven coverage for every documented Console event family and causal/privacy state. 2. Emit only missing transitions at existing runtime, controller, context, retrieval, and compaction owner seams. 3. Contain capture failures and persist an incomplete diagnostic when its existing owner remains writable. 4. Prove ordered capture through the real AgentService and repository seams with only the external provider faked. 5. Run focused compatibility, security/privacy, lint, and diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-080 capture contract across the existing Console message/trajectory sidecars, append-only AgentStep owner, retrieval/citation provenance, context operations, compaction attempts, provider gateway, and filesystem run-log owner. Conversation mutations; model request, streaming, retry, error, and cancellation; tool proposal, approval, execution, timeout, cancellation, and outcome; retrieval/context; and compaction observations now project as truthful causal events without a new event table or schema.

Agent lifecycle records use a persisted observation sequence separate from legacy control indices and budgets, with stable per-call identity and parent/source links. Capture failures remain best-effort and produce deterministic, restart-deduplicated `capture_failed` diagnostics when the existing owner is writable. Regeneration ancestry resolves through the active branch, provider fallback retries are recorded before dispatch, and legacy v1 Trace inputs retain their prior rendering/export behavior.

Privacy classification is applied at every durable boundary, including AgentRunsDB, Console tool markers, and decoded filesystem run logs. Credentials, hidden-reasoning encodings, and local path/file content are withheld with approved field states; safe content retains full-fidelity recovery handles. Structured inspection is single-pass, bounded, exception-safe, and preserves safe large content, URLs, routes, prose, and markup. Real-seam and adversarial tests cover joined order, reload identity, failure containment, active-path mutation visibility, privacy variants, recovery truthfulness, and 5,000-row diagnostic behavior. ADR: `backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md`.
<!-- SECTION:NOTES:END -->
