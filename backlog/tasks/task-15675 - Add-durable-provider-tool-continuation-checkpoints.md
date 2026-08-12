---
id: TASK-15675
title: Add durable provider tool-continuation checkpoints
status: In Progress
assignee: []
created_date: '2026-08-12 18:06'
labels: []
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-12-durable-provider-tool-continuation-design.md
  - >-
    Docs/superpowers/plans/2026-08-12-durable-provider-tool-continuation-implementation.md
  - backlog/decisions/058-hosted-provider-wire-and-durable-tool-continuation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Persist the private provider context required to resume interrupted native function-tool runs and satisfy documented later-turn reasoning replay without exposing reasoning in the transcript, re-executing completed tools, or creating a provider-specific agent loop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A versioned, bounded `provider_continuation_json` field is owned by the assistant generation/variant and stores only validated canonical continuation data, never credentials or raw provider responses.
- [ ] #2 The first complete assistant tool-call batch and its continuation checkpoint are durably created in one transaction before any tool executes; every call transition and provider-bound result is durable before the next provider request.
- [ ] #3 Restored `completed`/`failed` calls are never executed again, restored `executing` calls are treated as ambiguous and blocked, and pending calls require an explicit Resume action plus fresh approval.
- [ ] #4 Opening, importing, or syncing a conversation never starts tools automatically; resume pins the original provider, model, API mode, and normalized base while resolving the current credential normally.
- [ ] #5 Continuation data participates in message versioning, both supported sync paths, payload hashing, branch/variant ownership, deletion, edit/regenerate behavior, and whole-record conflict handling without field-level merge; the ChaChaNotes mutation and trigger-written intent commit together, then reconcile idempotently into the separate durable Sync-v2 outbox before configured portable tool execution.
- [ ] #6 Versioned `.chatbook` preserves/remaps graph and variant ownership before attaching private continuation, while ordinary active-path JSON uses an explicit private projection with a warning; text, Markdown, rendering, FTS, logs, errors, usage, and summaries exclude it.
- [ ] #7 Import validates version, provider, protocol, shapes, ordering, sizes, and call/result pairing; invalid private data is discarded with a safe warning while visible messages still import.
- [ ] #8 Provider-history expansion counts private continuation against the existing context budget and retains or evicts an owning visible turn and its private tool rounds atomically.
- [ ] #9 The shared runtime supports provider-specific replay policy without an open metadata bag: Kimi K3 replays retained reasoning on later K3 turns, other Kimi/GLM policies preserve only documented active/restored tool runs, and DeepSeek replays completed tool-associated reasoning on later same-provider turns.
- [ ] #10 Crash-boundary, sync/conflict, import/export, privacy, cancellation, branch/variant, and mutation tests prove the contract without paid requests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
The executable TDD plan is
`Docs/superpowers/plans/2026-08-12-durable-provider-tool-continuation-implementation.md`.
It is ordered as follows:

1. Define the bounded canonical checkpoint and owner-group contract without vendor wire translation.
2. Add schema-v36 message ownership and atomic persistence.
3. Extend Sync v1 and reconcile durable Sync-v2 outbox intent.
4. Add one typed runtime lifecycle callback before side effects.
5. Bind checkpoints to Console messages and explicit Resume/Discard UX.
6. Expand and budget provider-private history atomically with its visible owner.
7. Preserve/remap ownership in `.chatbook` and use explicit JSON projections.
8. Prove crash boundaries, privacy, conflicts, and closeout evidence.

ADR required: yes

ADR path: `backlog/decisions/058-hosted-provider-wire-and-durable-tool-continuation.md`

Reason: This task changes durable message storage, sync/export behavior, and the
provider/runtime side-effect boundary. ADR-058 records the approved contract.
<!-- SECTION:PLAN:END -->
