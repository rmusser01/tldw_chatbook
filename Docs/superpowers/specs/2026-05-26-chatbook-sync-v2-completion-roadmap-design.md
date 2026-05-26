# Chatbook Sync v2 Completion Roadmap

Date: 2026-05-26
Status: Draft implementation roadmap
Owner: `TASK-70`
Scope: Plan the remaining Sync v2 work from the current read-only/dry-run baseline to reliable manual local-first sync, then staged workspace/background expansion.

## Summary

Chatbook already has a substantial Sync v2 substrate: profile state, encrypted envelopes, restore services, recovery-key handling, outbox persistence, profile summary display, and read-only write-sync promotion labels. The remaining product gap is not another status banner. The missing step is a safe, user-visible path that produces real local changes into the Sync v2 outbox and lets a user manually sync them without silent mutation.

The first implementation milestone is intentionally small:

- Active server profile only.
- Personal dataset only.
- Manual sync only.
- Notes and Chat messages only.
- Explicit preview, run, conflict, restore, and recovery evidence before any background automation.

Workspace-scoped datasets, ACP package handoff, background sync, broad domain coverage, and collaboration remain later phases. They should reuse the same safety model after manual Notes and Chat sync proves reliable.

## Current Baseline

Completed foundations already provide these anchors:

- `TASK-15` through `TASK-59` established Sync v2 client dry-run, envelope, encryption, restore, recovery, outbox, runtime-policy, validation, and profile orchestration foundations.
- `TASK-59.1` added the Sync v2 profile summary contract.
- `TASK-60.4.2` shipped display-only write-sync promotion labels across relevant surfaces while explicitly not adding write replay, approval buttons, outbox drain, or server mutation.
- `TASK-69` surfaced read-only Sync v2 profile status in Library Collections.
- `Docs/superpowers/specs/2026-05-20-workspace-operating-context-handoff-prd-design.md` defines workspace visibility versus active-context eligibility and keeps background write sync deferred until manual push/pull is proven.

Current gap:

- User-created Notes and Chat message changes do not yet have a complete, product-approved path from local mutation to outbox envelope to manual push/pull result to restore verification.

## Product Principles

1. Sync must be source-honest. The UI must distinguish dry-run, staged outgoing work, pushed work, pulled work, conflict, blocked, and restored states.
2. Manual sync comes before automation. No scheduled/background mutation loop ships until users can safely preview, execute, recover, and audit manual sync.
3. Domain producers are explicit. Notes and Chat message changes must each own their outbox production contract instead of relying on a generic database diff.
4. Restore compatibility is part of the first milestone. A sync path is incomplete if a second device or repaired install cannot reconstruct the same Notes and Chat content.
5. Workspace switching does not hide Library or Notes items. Workspace-specific sync can come later; this milestone targets the active server profile personal dataset.
6. Conflict visibility beats silent resolution. Automatic resolution can be added only after the conflict object, user copy, retry path, and rollback evidence exist.

## First Milestone

### Included

- Notes outbox producer for user-created/edited/deleted note records in the active profile personal dataset.
- Chat message outbox producer for conversation/message mutations in the active profile personal dataset.
- Manual sync execution path that previews outgoing work, runs push/pull through existing Sync v2 services, and reports outcome.
- Conflict and partial-failure display states that reuse existing profile summary/status foundations.
- Restore/new-device verification for Notes and Chat messages.
- Actual app QA evidence for Settings, Library/Collections, Console, and any manual sync control surface touched.

### Excluded

- Background or scheduled sync.
- Workspace-scoped datasets.
- Real-time collaboration.
- ACP task/run package sync.
- MCP/tool runtime sync.
- Full Library/media/RAG/artifact domain sync.
- Silent merge or hidden conflict resolution.
- Server identity migration between unrelated profiles.

## Architecture Boundaries

### Domain Outbox Producers

Each included domain gets a small producer boundary:

| Domain | Producer responsibility | Non-responsibility |
| --- | --- | --- |
| Notes | Convert note create/update/delete into encrypted Sync v2 envelopes with stable local identity, domain scope, idempotency, and restore metadata. | UI layout, server transport, conflict review UI. |
| Chat | Convert conversation/message changes into ordered envelopes that preserve conversation identity, message parentage, variants, and restore continuity. | Chat rendering, LLM streaming, agent runtime orchestration. |

Producer requirements:

- Run inside the same local transaction boundary as the user-visible mutation when feasible, or record a durable recovery marker if enqueue happens after the mutation.
- Use stable `client_envelope_id` and domain-qualified `source_scope_key` values.
- Avoid exposing plaintext in logs, status labels, or task notes.
- Keep local-first success independent of server availability.
- Produce deterministic dry-run evidence for tests.

### Manual Sync Execution

Manual sync is a user-controlled workflow:

1. Preflight reads profile summary and pending domain counts.
2. Preview shows outgoing/incoming/conflict expectations using safe labels.
3. User explicitly starts sync.
4. Push runs first for retained outgoing envelopes.
5. Pull applies accepted remote envelopes only after local validation passes.
6. Status updates show success, partial success, conflict, blocked, or failed.
7. Failed or conflicted outgoing work remains visible and retryable.

Manual sync must not:

- Auto-run on app launch.
- Hide partial push failures after a successful pull.
- Advance cursors after failed apply.
- Convert dry-run labels into mutation controls before the execution path exists.

### Conflict And Recovery

Conflict review needs to expose:

- Domain.
- Item label.
- Local version summary.
- Remote version summary.
- Cause.
- Safe user actions: keep local, accept remote, duplicate/fork, retry later, or inspect raw audit where available.

For the first milestone, a conflict can block completion as long as it is durable, visible, and recoverable. A polished merge editor is not required before the first manual sync path lands.

### Restore Compatibility

Restore/new-device validation must prove:

- Notes restored from synced envelopes retain title/body/status/deletion semantics.
- Chat conversations restore message order, roles, parentage, and variants needed for resumed conversation continuity.
- Recovery-key restore can decrypt selected envelopes without persisting recovered secrets inappropriately.
- Restore failure stops before applying partial corrupt state.

## Phased Plan

### Phase 0: Roadmap And Task Split

Deliver this spec and Backlog child tasks. No runtime behavior changes.

Exit evidence:

- Spec is committed.
- Child tasks are created with PR-sized acceptance criteria.
- Current baseline and exclusions are explicit.

### Phase 1: Notes Producer

Implement the first content-producing path for Notes.

Exit evidence:

- Note create/update/delete can enqueue retained Sync v2 envelopes.
- Local note operations still succeed when no server is reachable.
- Tests prove envelope shape, idempotency, encryption boundary, and profile summary pending counts.

### Phase 2: Chat Producer

Implement Chat conversation/message outbox production.

Exit evidence:

- User and assistant messages can enqueue ordered Sync v2 envelopes.
- Conversation identity, message order, variants, and restore metadata are preserved.
- Streaming or failed-send paths do not enqueue misleading complete-message envelopes.

### Phase 3: Manual Sync Control Surface

Add the manual sync preview/run/result path.

Exit evidence:

- Users can preview pending Notes/Chat work and explicitly run sync.
- UI reports success, partial success, conflict, blocked, and failure states.
- No background sync is introduced.

### Phase 4: Conflict Review And Recovery

Make conflict states actionable enough for manual use.

Exit evidence:

- Conflicts are listed with domain/item/cause.
- Users have clear recovery actions.
- Partial push failures remain visible until resolved.

### Phase 5: Restore/New-Device QA

Validate that synced Notes and Chat messages survive restore.

Exit evidence:

- A clean local profile can restore selected Notes and Chat envelopes.
- Recovery-key flow is covered.
- Actual app QA verifies users can understand what was restored and what failed.

### Phase 6: Manual Sync Closeout Gate

Run an actual-use QA pass before considering broader promotion.

Exit evidence:

- Manual Notes+Chat sync works end-to-end against a local/server test target.
- Settings, Library/Collections, and Console agree on status.
- Remaining gaps are filed as separate tasks.

### Later Phases

Later work should only start after the manual closeout gate:

- Workspace-scoped datasets and eligibility-aware sync.
- ACP task/run package sync and handoff.
- Library/media/RAG/artifact domain producers.
- Scheduled/background sync.
- Collaboration/shared workspace semantics.

## Backlog Child Tasks

This roadmap is split into these immediate child tasks:

- `TASK-70.1`: Add Notes Sync v2 outbox producer plan.
- `TASK-70.2`: Add Chat Sync v2 outbox producer plan.
- `TASK-70.3`: Add manual Sync v2 preview and execution plan.
- `TASK-70.4`: Add Sync v2 conflict review and recovery plan.
- `TASK-70.5`: Add Sync v2 Notes and Chat restore QA plan.
- `TASK-70.6`: Close out manual Sync v2 milestone with actual-use QA.

Each child task should be implemented in its own PR unless review shows two adjacent slices are too small to validate independently.

## Verification Strategy

Every implementation child task should use:

- A failing regression before implementation.
- Focused unit tests for pure envelope/status logic.
- Integration tests for local database, outbox, transport, and restore behavior.
- Mounted Textual tests for visible control states.
- Actual app QA through CDP/textual-web screenshots for any visible Settings, Library, Console, or Home surface.
- `git diff --check` before PR.

Recommended common commands:

```bash
python -m pytest -q Tests/Sync_Interop --tb=short
python -m pytest -q Tests/UI/test_product_maturity_phase1_harness.py::test_backlog_task_frontmatter_ids_are_unique --tb=short
git diff --check
```

## Risks

- Notes and Chat may have different local transaction seams, so a shared producer abstraction could become too generic too early.
- Chat streaming can create partial message states; producers must avoid syncing incomplete assistant messages as final content.
- Backlog task IDs can drift during parallel PR work. Roadmap tasks should run the frontmatter uniqueness guard before merge.
- Manual sync UI could accidentally imply background sync is enabled. Copy and control labels must remain explicit.
- Restore validation can pass technically while still being confusing to users. Actual-use QA is required before milestone closeout.

## Open Questions

- Which server test target should be canonical for manual sync closeout: local `tldw_server`, `tldw_server2`, or a mocked Sync v2 API harness?
- Should Notes delete sync as tombstones only, or should the first milestone support undelete/recovery metadata?
- Which Chat message variant model is canonical for restore when regenerated assistant messages exist?
- Where should the primary manual sync action live first: Settings Sync Safety, Library Collections, Home status, or a dedicated Sync modal?
- What audit detail is safe to expose by default without leaking note/chat plaintext?

## Approval Gate

This roadmap is approved for implementation only when:

- The user accepts the milestone scope and exclusions.
- Child tasks are reviewed as PR-sized.
- The first child task starts with Notes producer tests rather than UI polish.
