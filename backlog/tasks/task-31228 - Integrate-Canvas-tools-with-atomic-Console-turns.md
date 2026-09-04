---
id: TASK-31228
title: Integrate Canvas tools with atomic Console turns
status: To Do
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-03'
labels: [canvas, agents, console]
dependencies: [TASK-31226, TASK-31227]
priority: high
---

## Description

Expose explicit conversation-scoped Canvas tools to Console assistants and make their staged revisions commit or disappear with the originating assistant turn, without copying sensitive HTML into generic tool records, logs, diagnostics, or transcript cards.

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

## Related Design

- `Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md`
- `Docs/superpowers/plans/2026-09-03-chatbook-canvas-implementation.md`
- `backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md`
