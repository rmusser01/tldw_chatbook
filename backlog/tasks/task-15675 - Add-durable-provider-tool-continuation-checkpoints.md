---
id: TASK-15675
title: Add durable provider tool-continuation checkpoints
status: Done
assignee: []
created_date: '2026-08-12 18:06'
updated_date: '2026-08-13 17:58'
labels: []
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-12-durable-provider-tool-continuation-design.md
  - >-
    Docs/superpowers/plans/2026-08-12-durable-provider-tool-continuation-implementation.md
  - backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Persist the private provider context required to resume interrupted native function-tool runs and satisfy documented later-turn reasoning replay without exposing reasoning in the transcript, re-executing completed tools, or creating a provider-specific agent loop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A versioned, bounded `provider_continuation_json` field is owned by the assistant generation/variant and stores only validated canonical continuation data, never credentials or raw provider responses.
- [x] #2 The first complete assistant tool-call batch and its continuation checkpoint are durably created in one transaction before any tool executes; every call transition and provider-bound result is durable before the next provider request.
- [x] #3 Restored `completed`/`failed` calls are never executed again, restored `executing` calls are treated as ambiguous and blocked, and pending calls require an explicit Resume action plus fresh approval.
- [x] #4 Opening, importing, or syncing a conversation never starts tools automatically; resume pins the original provider, model, API mode, and normalized base while resolving the current credential normally.
- [x] #5 Continuation data participates in message versioning, both supported sync paths, payload hashing, branch/variant ownership, deletion, edit/regenerate behavior, and whole-record conflict handling without field-level merge; the ChaChaNotes mutation and trigger-written intent commit together, then reconcile idempotently into the separate durable Sync-v2 outbox before configured portable tool execution.
- [x] #6 Versioned `.chatbook` preserves/remaps graph and variant ownership before attaching private continuation, while ordinary active-path JSON uses an explicit private projection with a warning; text, Markdown, rendering, FTS, logs, errors, usage, and summaries exclude it.
- [x] #7 Import validates version, provider, protocol, shapes, ordering, sizes, and call/result pairing; invalid private data is discarded with a safe warning while visible messages still import.
- [x] #8 Provider-history expansion counts private continuation against the existing context budget and retains or evicts an owning visible turn and its private tool rounds atomically.
- [x] #9 The shared runtime supports provider-specific replay policy without an open metadata bag: Kimi K3 replays retained reasoning on later K3 turns, other Kimi/GLM policies preserve only documented active/restored tool runs, and DeepSeek replays completed tool-associated reasoning on later same-provider turns.
- [x] #10 Crash-boundary, sync/conflict, import/export, privacy, cancellation, branch/variant, and mutation tests prove the contract without paid requests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
The executable TDD plan is
`Docs/superpowers/plans/2026-08-12-durable-provider-tool-continuation-implementation.md`.
It is ordered as follows:

1. Define the bounded canonical checkpoint and owner-group contract without vendor wire translation.
2. Add schema-v37 message ownership and atomic persistence.
3. Extend Sync v1 and reconcile durable Sync-v2 outbox intent.
4. Add one typed runtime lifecycle callback before side effects.
5. Bind checkpoints to Console messages and explicit Resume/Discard UX.
6. Expand and budget provider-private history atomically with its visible owner.
7. Preserve/remap ownership in `.chatbook` and use explicit JSON projections.
8. Prove crash boundaries, privacy, conflicts, and closeout evidence.

ADR required: yes

ADR path: `backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md`

Reason: This task changes durable message storage, sync/export behavior, and the
provider/runtime side-effect boundary. ADR-063 records the approved contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the bounded canonical continuation record on assistant message owners,
  schema-v37 persistence and migration, both sync paths plus durable scoped
  Sync-v2 projection receipts, typed runtime lifecycle barriers, explicit
  Console Resume/Take over/Discard recovery, atomic history budgeting, and
  versioned `.chatbook`/explicit JSON portability. ADR-063 remains the governing
  storage, sync, runtime, privacy, and provider-policy boundary; provider wire
  adapters remain follow-up work.
- Added deterministic joined crash coverage for all seven approved boundaries
  and mutation/property coverage for terminal/ambiguous replay, validation
  bounds and unknown versions, whole-record branch/variant conflicts, and
  atomic owner/private eviction. Final review replaced harness-only boundaries
  with production runtime/store crash hooks and restart assertions. No paid or
  live provider request is present.
- Final review corrections project every lifecycle and terminal mutation from
  its exact committed whole-message intent, propagate inbound deletes, and
  reconcile exact current upsert/delete intents (including continuation clear
  and owner deletion) during production restore. Provider-policy-specific prior
  continuation sidecars now pass through the gateway's ordinary history budget:
  Kimi K3 and DeepSeek retain their documented prior history, while other
  Kimi/GLM policies do not receive unrelated prior sidecars. These corrections
  close AC5, AC9, and AC10 without adding provider wire adapters or an open
  metadata bag.
- Post-rebase focused continuation verification: `384 passed, 2 warnings in
  37.61s` across 13 named canonical storage/migration, runtime/eviction,
  Console persistence/recovery/history/privacy/budget, Sync-v2 reconciliation,
  and `.chatbook` round-trip files. The schema-v36 note-folder plus schema-v37
  continuation migration pair passed `14 passed`; selected joined
  controller/gateway/bridge/service/store paths passed `14 passed`.
- Settled touched surfaces: Agents `1386 passed`; Chat effective `4536 passed,
  3 verified baseline failures, 64 skipped`; Chatbooks `243 passed, 1 skipped`;
  Sync Interop `250 passed`; DB retained 37 verified pre-task failures after
  three branch-caused fixture failures were fixed and their focused 14 tests
  passed; ChaChaNotesDB retained 2 verified baseline failures with 187 passed.
- A full-repository pytest run reached 86% before the user explicitly stopped
  broad testing. It has no terminal summary, was not restarted, and is not used
  as passing evidence; the user directed closeout from directly related tests
  only.
- Qodo follow-up made continuation-aware eviction fail closed with a sanitized
  frame trace, added safe import operation/source/category context, and routed
  continuation sync/tombstone reads through the shared transaction boundary
  with complete public docstrings. The focused review matrix passed `79 tests`.
  Qodo's persistence-hook compatibility suggestion was declined: ADR-063
  requires persistent conversations to fail closed before tool side effects,
  while explicitly ephemeral contexts already remain non-resumable.
- Static evidence: the rebase-resolution Python files pass Ruff lint and
  compileall, the changed migration test passes Ruff format, and diff checks
  pass. Whole-file Ruff formatting still reports three large legacy production
  modules; all three deviations reproduce on rebased `origin/dev`, so no
  formatter-only churn was introduced.
- Documentation explains private-state exclusions, explicit recovery and
  ambiguity, frozen targets with current credential resolution, local commit
  plus idempotent encrypted Sync-v2 projection, export compatibility, and the
  absence of provider-retention or cross-device exactly-once claims.
- Changed areas: canonical Chat continuation and history, ChaChaNotes storage,
  Sync Interop, Agents runtime, Console recovery, Chatbook import/export,
  migrations, focused tests, README/user guides, this plan/task, and ADR/spec
  references. No provider adapter/default, vendor built-in tool, legacy
  Settings surface, or paid test was added.
<!-- SECTION:NOTES:END -->
