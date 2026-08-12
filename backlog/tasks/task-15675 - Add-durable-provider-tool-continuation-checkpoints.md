---
id: TASK-15675
title: Add durable provider tool-continuation checkpoints
status: In Progress
assignee: []
created_date: '2026-08-12 18:06'
labels: []
dependencies: []
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
- [ ] #5 Continuation data participates in message versioning, both supported sync paths, payload hashing, branch/variant ownership, deletion, edit/regenerate behavior, and whole-record conflict handling without field-level merge; when sync is enabled, each pre-dispatch checkpoint and its durable outbox intent commit together.
- [ ] #6 Versioned `.chatbook` preserves/remaps graph and variant ownership before attaching private continuation, while ordinary active-path JSON uses an explicit private projection with a warning; text, Markdown, rendering, FTS, logs, errors, usage, and summaries exclude it.
- [ ] #7 Import validates version, provider, protocol, shapes, ordering, sizes, and call/result pairing; invalid private data is discarded with a safe warning while visible messages still import.
- [ ] #8 Provider-history expansion counts private continuation against the existing context budget and retains or evicts an owning visible turn and its private tool rounds atomically.
- [ ] #9 The shared runtime supports provider-specific replay policy without an open metadata bag: Kimi K3 replays retained reasoning on later K3 turns, other Kimi/GLM policies preserve only documented active/restored tool runs, and DeepSeek replays completed tool-associated reasoning on later same-provider turns.
- [ ] #10 Crash-boundary, sync/conflict, import/export, privacy, cancellation, branch/variant, and mutation tests prove the contract without paid requests.
<!-- AC:END -->
