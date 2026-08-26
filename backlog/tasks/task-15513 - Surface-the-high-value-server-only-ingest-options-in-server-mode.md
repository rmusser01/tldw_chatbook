---
id: TASK-15513
title: Expose high-value ingest options with local parity
status: Done
assignee:
  - '@codex'
created_date: ''
updated_date: '2026-08-13 02:29'
labels:
  - library
  - ingest
  - parity
  - server
  - local
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-3309 closed request-layer server parity. This task exposes overwrite_existing, custom_prompt, system_prompt, and generate_embeddings as shared controls with real Local and Server behavior. keep_original_file remains Server-only: Local ingestion already leaves the source file untouched, and this task does not create a managed-copy/archive feature. The remaining server-declared fields from the live audit remain deliberately unexposed. The canvas must never display a control that the selected backend silently ignores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared controls for overwrite existing, custom prompt, system prompt, and generate embeddings are visible and settable in both Local and Server modes.
- [x] #2 Each shared control reaches the selected backend and has focused tests proving its local pipeline behavior and server request mapping; no selection is silently inert.
- [x] #3 Custom and system prompt controls clearly communicate and enforce their relationship to analysis, including a readable disabled reason when analysis is off.
- [x] #4 Local overwrite updates the matching Library item according to the local deduplication contract, while Server overwrite sends overwrite_existing to the declared server field.
- [x] #5 Local generate embeddings invokes the supported local embedding and indexing path, while Server generate embeddings sends generate_embeddings to the declared server field.
- [x] #6 keep_original_file is available and functional in Server mode, reaches the declared server field, and is not rendered in Local mode because local ingestion already preserves the source file.
- [x] #7 The remaining server-only fields stay recorded as deliberately unexposed, with no unsupported controls shown in either mode.
<!-- AC:END -->

## Implementation Plan

1. Extend the capability-driven Import behavior panel with backend visibility metadata, four shared controls, and the Server-only Keep original file control. Preserve keyboard reachability, compact geometry, state receipts, reset, retry, and config persistence.
2. Project the shared generic options explicitly into both Local and Server backend boundaries. Gate analysis prompts when Analyze after import is off and assert Server fields against the captured declared-field fixture.
3. Route Local overwrite to the authoritative SQLite dedup/update seam and prove the observable duplicate outcome changes when selected.
4. Add a context-local opt-out around the existing best-effort RAG post-ingest hook so Generate embeddings off affects only that Local import without changing global settings or source persistence.
5. Run focused behavior, rendered-frame, static, mutation, and backlog checks; document evidence and close the task only when every acceptance criterion is verified.

ADR required: no new ADR.

ADR paths:

- `backlog/decisions/005-invest-in-local-rag-mirroring-tldw-server.md`
- `backlog/decisions/030-derived-index-lifecycle-and-atomic-media-migrations.md`

Reason: the accepted ADRs already define default local ingestion-time indexing and the authoritative-source/derived-index lifecycle. This task adds per-import UI and routing within those boundaries.

Detailed design: `Docs/superpowers/specs/2026-08-12-task-15513-ingest-option-local-parity-design.md`.

Detailed plan: `Docs/superpowers/plans/2026-08-12-task-15513-ingest-option-local-parity.md`.

## Implementation Notes

- Extended the capability-driven Import behavior panel with backend visibility metadata, multiline analysis prompts, shared overwrite/indexing controls, and Server-only Keep original file. Textarea edits update in place, preserve focus/cursor state, and use bounded collapsed-title receipts.
- Seeded fresh snapshots from capability defaults and preserved all values through retry. Local and Server projections explicitly gate prompts on Analyze after import; Server requests map every declared shared field while Local never receives Keep original file.
- Routed Local overwrite to the authoritative SQLite dedup/update seam and proved Off skips while On updates the existing row. Added a token-reset `ContextVar` scope that suppresses only the best-effort semantic indexing hook for a Local ingest when Generate embeddings is off; SQLite persistence remains unconditional and concurrent threads remain isolated.
- Rebasing onto the latest `dev` incorporated the Library canvas-scoped-sync refactor. Structural controls retain canvas-scoped synchronization, while text, number, and textarea edits remain non-recomposing.
- Verification on the rebased branch: all 22 new and integration-focused tests passed, including normal/constrained full-screen compositor geometry, retry/server contract coverage, real SQLite overwrite, default-On and opt-out indexing, exception reset, thread isolation, and upstream ingest canvas-sync behavior. Python compilation, isolated import smoke, `git diff --check`, the 1,852-task duplicate-ID guard, and changed-line Ruff checks passed. Ruff still reports 148 pre-existing whole-file findings outside added/modified lines; broad Windows Textual/RAG runs remain limited by the repository network guard blocking Proactor socketpair setup.
- Plan evidence deviation: full-screen compositor assertions at 170x48 and constrained 120x48 replaced a persisted screenshot at 99 columns because the production Library frame allocates a zero-width canvas at that narrower size. TDD RED runs and mutation-sensitive regressions cover the backend filter, prompt gate, overwrite forwarding, schema-default delegation, and indexing suppression guard.
- ADR required: no new ADR. The implementation follows ADR-005 for default ingestion-time indexing and ADR-030 for authoritative SQLite persistence with a best-effort derived semantic index.
