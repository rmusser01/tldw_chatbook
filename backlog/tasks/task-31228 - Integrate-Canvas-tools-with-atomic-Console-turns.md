---
id: TASK-31228
title: Integrate Canvas tools with atomic Console turns
status: Done
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-04 14:09'
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
- [x] #1 `canvas_list`, `canvas_read`, `canvas_create`, and `canvas_update` are advertised only for enabled Console sessions with injected conversation/run/branch scope
- [x] #2 Tool inputs and outputs implement the approved full-document and optimistic-parent contracts with bounded compatibility errors
- [x] #3 Canvas mutations are reversible local operations and bypass ordinary tool approval without weakening approval behavior for any other tool
- [x] #4 Successful tool mutations stage idempotently by session/run/tool-call identity and remain visibly uncommitted until turn finalization
- [x] #5 The assistant message, Canvas transcript-card metadata, and every staged revision commit in one transaction; cancellation or terminal failure discards the stage
- [x] #6 Parallel same-Canvas mutation batches refuse ambiguity while sequential same-turn updates preserve ancestry
- [x] #7 Model, invocation, display, log, cycle-detection, and continuation projections expose exactly the approved Canvas fields and keep HTML out of generic durable records
- [x] #8 A Canvas-only assistant turn still creates the message/turn anchor required by every committed revision
- [x] #9 Focused provider, runtime, projection, commit-failure, cancellation, concurrency, and resume tests pass
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented audience-specific tool projections, exact scoped Canvas tools, reversible Canvas-only approval classification, and run-owned staging integrated with production Console composition. Assistant text, Canvas card metadata, revisions, and checkpoint settlement now share the existing transaction for both initial and retried assistant anchors; explicit empty metadata clears stale retry cards, while true database failures retain READY stages. Temporary sessions use incarnation/run/promotion leases, branch-authorized reads, atomic promotion with durable origin remapping, and exact teardown. Independent review completed three correction rounds and approved the final result with no Critical, Important, or Minor findings. Verification: 767 targeted tests passed; Python compilation, changed-file fatal Ruff excluding the reproduced pre-existing FallbackRuntime F821, and git diff checks passed. No full suite was run under repository policy. ADR required: yes; implemented existing ADR-115 without a new ADR.
<!-- SECTION:NOTES:END -->

## Delivery 3 checkpoint (Task 3.3)

- Projection inventory: model history/invocation receives validated source only where the Canvas tool contract requires it; display, run-log, cycle, continuation, AgentStep, and transcript projections contain bounded IDs/title/sequence/digest/status/origin metadata only. Transcript cards reopen source through the Canvas service rather than storing it in generic message metadata.
- Transaction boundary: the authenticated provider scope's server-issued run ID is used as the actual primary Agent row ID. Initial and retried assistant anchors update message content, explicitly replaced Canvas-card metadata, and staged document/revision contributions in their existing caller-owned SQLite transactions; the initial path also deletes the dispatch checkpoint in that transaction. The controller marks the stage committed only after transaction success. A write failure rolls back message and Canvas rows and leaves the READY stage eligible for bounded retry; omitted metadata remains distinct from an explicit empty replacement so a zero-Canvas retry clears stale failed-attempt cards without erasing unrelated metadata.
- Cancellation and lifecycle: provider error, cancellation, stuck runs, run-ID mismatch, session close, state replacement, and runtime shutdown fail closed and discard exact run-owned source once. Duplicate terminal callbacks reuse the frozen settlement, and duplicate `(run_id, tool_call_id)` calls reuse the staged revision.
- Concurrency and resume: sequential same-Canvas calls must name the preceding staged revision; a second same-parent call receives bounded `ambiguous_ancestry` without mutation. Continuation/re-entry returns the existing frozen settlement and does not duplicate revisions.
- Sentinel evidence: `test_every_non_model_projection_omits_canvas_source`, the adversarial projection tests, and the real Agent review-batch persistence test prove the source sentinel is absent from display/log/cycle/continuation projections and serialized Agent rows. `test_metadata_serialization_never_contains_source` and `test_transcript_restores_metadata_only_canvas_card` prove the turn settlement, restored transcript card, and plain-text transcript remain source-free.
- ADR required: no new ADR. This delivery directly implements the accepted ownership, privacy, and transaction boundaries in ADR-115.
- Review status: independent review covered transaction ordering, production composition, exact session/run/promotion fencing, retry ownership, canonical repository enforcement, branch authority, approval-bypass scope, and source leakage. Three correction rounds resolved all reported findings; final review approved the delivery with no Critical, Important, or Minor issues.
