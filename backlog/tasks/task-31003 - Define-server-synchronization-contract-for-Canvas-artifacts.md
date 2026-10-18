---
id: TASK-31003
title: Define server synchronization contract for Canvas artifacts
status: In Progress
assignee:
- '@codex'
created_date: 2026-09-03 13:39
updated_date: 2026-10-18 05:01
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make durable Canvas documents and immutable revisions portable across devices without weakening conversation ownership, branch semantics, deletion lifecycle, or the sandbox security boundary. This follows the local-first Canvas release, whose export/import path provides initial portability.
<!-- SECTION:DESCRIPTION:END -->

## Related Design

- `Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md`
- `backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md`
- `backlog/decisions/081-local-versioned-canvas-artifacts-and-browser-sandbox.md`
- `backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md`

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Canvas sync ownership and service contract are documented and approved
- [ ] #2 Only canvases belonging to synchronized conversations can synchronize
- [ ] #3 Stable canvas identity, revision ancestry, origin-message linkage, titles, and deletion state round-trip without loss
- [ ] #4 Active-branch projection resolves to the same reachable revision after synchronization
- [ ] #5 Concurrent updates and delete-versus-update conflicts have deterministic user-visible outcomes
- [ ] #6 Older clients preserve or safely ignore unsupported Canvas data without corrupting conversations
- [ ] #7 Canvas HTML never executes during synchronization, and sync grants no browser or runtime capability
- [ ] #8 Automated tests cover create, update, delete, restore, conflict, retry, idempotency, and mixed-version behavior
- [ ] #9 Relevant ADR, sync documentation, and migration notes are updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/239-canvas-sync-replica-and-lifecycle-contract.md (Proposed)
Reason: new synchronization ownership, replica ordering, deletion/conflict and compatibility contracts require explicit approval before implementation.
1. Inspect actual Canvas storage/archives and existing Sync v2 transport and companion-server source; distinguish supported capabilities from proposals.
2. Prepare Docs/superpowers/specs/2026-10-17-canvas-server-sync-contract-design.md with wire examples, deterministic conflicts, negotiation/limits, security boundaries and a concrete conformance matrix.
3. Review storage sequence collisions, archive compatibility, source readability and server dependencies with the user; retain all current acceptance criteria open until their stated evidence exists.
4. After approval, separate design approval from coordinated storage/archive, server adapters and client integration/qualification work using atomic tasks. Do not activate synchronization during this design task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-17 design progress only: inspected actual Canvas storage, archives and Sync v2 plus companion-server source at tldw_server 204a13724d0484eecb942d16fe82a28f08e68346 and tldw_server2 81258987760de25161fa4eb050c725e0b4618080. Neither server has Canvas sync domains. Prepared the concrete reviewable wire/lifecycle/compatibility proposal in Docs/superpowers/specs/2026-10-17-canvas-server-sync-contract-design.md and Proposed ADR239. Current per-canvas sequence uniqueness cannot preserve both offline siblings; the 256 KiB default envelope cannot carry every accepted source. The proposal documents future migration/archive compatibility, server-readable source consent, deterministic retained conflicts, delete/restore fences, complete capability negotiation, idempotency and twelve conformance cases. No sync, schema, archive or runtime capability is activated. All original criteria remain open: user approval of the contract/scope and coordinated server implementation are still required. Do not treat design text as implementation evidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
