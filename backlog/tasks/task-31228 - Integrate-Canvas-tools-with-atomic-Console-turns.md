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
