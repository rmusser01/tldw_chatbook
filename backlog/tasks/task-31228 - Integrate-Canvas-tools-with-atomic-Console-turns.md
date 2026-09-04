---
id: TASK-31228
title: Integrate Canvas tools with atomic Console turns
status: In Progress
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-04 09:40'
labels:
  - canvas
  - agents
  - console
dependencies:
  - TASK-31226
  - TASK-31227
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose explicit conversation-scoped Canvas tools to Console assistants and make their staged revisions commit or disappear with the originating assistant turn, without copying sensitive HTML into generic tool records, logs, diagnostics, or transcript cards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `canvas_list`, `canvas_read`, `canvas_create`, and `canvas_update` are advertised only for enabled Console sessions with injected conversation/run/branch scope
- [ ] #2 Tool inputs and outputs implement the approved full-document and optimistic-parent contracts with bounded compatibility errors
- [ ] #3 Canvas mutations are reversible local operations and bypass ordinary tool approval without weakening approval behavior for any other tool
- [ ] #4 Successful tool mutations stage idempotently by session/run/tool-call identity and remain visibly uncommitted until turn finalization
- [ ] #5 The assistant message, Canvas transcript-card metadata, and every staged revision commit in one transaction; cancellation or terminal failure discards the stage
- [ ] #6 Parallel same-Canvas mutation batches refuse ambiguity while sequential same-turn updates preserve ancestry
- [ ] #7 Model, invocation, display, log, cycle-detection, and continuation projections expose exactly the approved Canvas fields and keep HTML out of generic durable records
- [ ] #8 A Canvas-only assistant turn still creates the message/turn anchor required by every committed revision
- [ ] #9 Focused provider, runtime, projection, commit-failure, cancellation, concurrency, and resume tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md
Reason: this task implements ADR-115’s agent projection, scoped tool authority, approval classification, and assistant-turn transaction boundaries; no new ADR is needed unless implementation changes those accepted security or ownership contracts.

1. Inventory every raw tool argument/result consumer and add a generic audience-specific projection seam whose default preserves all existing providers while failures redact closed.
2. Register the four Canvas tools behind enabled Console session scope, inject all authority fields, enforce shared limits/full-document optimistic contracts, and narrowly pre-authorize only reversible Canvas mutations.
3. Coordinate run-owned Canvas staging so successful assistant messages, card metadata, and revisions commit atomically, while cancellation/failure discards exact staged state and continuation remains source-free.
4. Run focused Agent catalog/runtime/projection, approval, Console controller/persistence/cancellation/continuation, and transcript suites plus static checks.
5. Request independent review focused on approval-bypass scope and source leakage, then update TASK-31228 and the implementation plan with transaction, cancellation, inventory, and sentinel evidence.
<!-- SECTION:PLAN:END -->

## Delivery 3 checkpoint (Task 3.3)

- Projection inventory: model history/invocation receives validated source only where the Canvas tool contract requires it; display, run-log, cycle, continuation, AgentStep, and transcript projections contain bounded IDs/title/sequence/digest/status/origin metadata only. Transcript cards reopen source through the Canvas service rather than storing it in generic message metadata.
- Transaction boundary: the authenticated provider scope's server-issued run ID is used as the actual primary Agent row ID. The already-created assistant anchor is then updated to its terminal state, Canvas card metadata is written on that message, and the staged document/revision contribution executes through the same caller-owned `ConsoleDispatchRepository.settle_with_assistant` transaction before the dispatch checkpoint is deleted. The controller marks the stage committed only after that transaction reports success; a write failure rolls back message and Canvas rows and leaves the READY stage eligible for bounded retry.
- Cancellation and lifecycle: provider error, cancellation, stuck runs, run-ID mismatch, session close, state replacement, and runtime shutdown fail closed and discard exact run-owned source once. Duplicate terminal callbacks reuse the frozen settlement, and duplicate `(run_id, tool_call_id)` calls reuse the staged revision.
- Concurrency and resume: sequential same-Canvas calls must name the preceding staged revision; a second same-parent call receives bounded `ambiguous_ancestry` without mutation. Continuation/re-entry returns the existing frozen settlement and does not duplicate revisions.
- Sentinel evidence: `test_every_non_model_projection_omits_canvas_source`, the adversarial projection tests, and the real Agent review-batch persistence test prove the source sentinel is absent from display/log/cycle/continuation projections and serialized Agent rows. `test_metadata_serialization_never_contains_source` and `test_transcript_restores_metadata_only_canvas_card` prove the turn settlement, restored transcript card, and plain-text transcript remain source-free.
- ADR required: no new ADR. This delivery directly implements the accepted ownership, privacy, and transaction boundaries in ADR-115.
- Review status: implementation was self-reviewed for transaction ordering, approval-authority reach, source-bearing object representations, lifecycle cleanup, and retry behavior. Independent subagent review was intentionally not launched because the assigned task explicitly prohibited spawning reviewers; parent review remains available at integration.
