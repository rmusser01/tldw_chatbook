---
id: TASK-31003
title: Define server synchronization contract for Canvas artifacts
status: Done
assignee:
- '@codex'
created_date: 2026-09-03 13:39
updated_date: 2026-10-18 06:03
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
- [x] #1 Canvas sync ownership and the explicit target service contract are documented in the user-approved spec and accepted ADR-239; approval is not represented as server compatibility.
- [x] #2 The contract specifies durable synchronized-conversation eligibility, complete message-tree ownership and exclusion of Temporary or uncommitted work.
- [x] #3 The contract specifies lossless stable identity, immutable source/profile/hash/metadata, revision ancestry, origin linkage, titles and lifecycle wire records.
- [x] #4 The contract defines deterministic reachable-revision projection while keeping active branch, browser selections and execution grants local.
- [x] #5 The contract specifies user-visible sibling and immutable-identity conflicts, idempotency, delete-dominant visibility, explicit restore epochs and retired-owner barriers.
- [x] #6 The contract defines mixed-version, unsupported-capability, upgrade-baseline and older-client behavior without conversation or Canvas data loss.
- [x] #7 The contract explicitly prohibits HTML execution and capability transfer during sync, documents source-free diagnostics and discloses authenticated-home-server readability.
- [x] #8 A concrete implementation conformance matrix covers create/update/rename, delete/restore/purge, conflict, retry/idempotency, crashes, limits and mixed-version behavior; it does not claim unrun feature tests.
- [x] #9 ADR, living architecture/sync documentation and migration/archive compatibility requirements are updated, and the approved design maps remaining implementation and real-server qualification to atomic follow-up deliverables.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/239-canvas-sync-replica-and-lifecycle-contract.md (Accepted; user approved 2026-10-17)
Reason: new synchronization ownership, replica ordering, deletion/conflict and compatibility contracts require explicit approval before implementation.
1. Inspect actual Canvas storage/archives and existing Sync v2 transport and companion-server source; distinguish supported capabilities from proposals.
2. Prepare Docs/superpowers/specs/2026-10-17-canvas-server-sync-contract-design.md with wire examples, deterministic conflicts, negotiation/limits, security boundaries and a concrete conformance matrix.
3. Review storage sequence collisions, archive compatibility, source readability and server dependencies with the user; retain all current acceptance criteria open until their stated evidence exists.
4. After approval, separate design approval from coordinated storage/archive, server adapters and client integration/qualification work using atomic tasks. Do not activate synchronization during this design task.
2026-10-17 user approved the written ADR239 contract and splitting client/server implementation into atomic follow-ups. Amend this task to contract-design outcomes, preserve the original feature acceptance coverage in the design delivery map, and qualify the first inert client foundation separately. Server repository adoption and actual interoperability remain implementation gates, not a design-completion claim.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-17 design progress only: inspected actual Canvas storage, archives and Sync v2 plus companion-server source at tldw_server 204a13724d0484eecb942d16fe82a28f08e68346 and tldw_server2 81258987760de25161fa4eb050c725e0b4618080. Neither server has Canvas sync domains. Prepared the concrete reviewable wire/lifecycle/compatibility proposal in Docs/superpowers/specs/2026-10-17-canvas-server-sync-contract-design.md and Proposed ADR239. Current per-canvas sequence uniqueness cannot preserve both offline siblings; the 256 KiB default envelope cannot carry every accepted source. The proposal documents future migration/archive compatibility, server-readable source consent, deterministic retained conflicts, delete/restore fences, complete capability negotiation, idempotency and twelve conformance cases. No sync, schema, archive or runtime capability is activated. All original criteria remain open: user approval of the contract/scope and coordinated server implementation are still required. Do not treat design text as implementation evidence.
2026-10-17 user approved the written contract and explicitly requested the atomic client/server follow-up split. ADR239 is Accepted for Chatbook implementation, with server adoption/qualification still outstanding. Acceptance criteria were amended with that explicit approval from full-feature delivery to contract-design outcomes; original implementation coverage is preserved in the spec delivery map. The approved design covers exact domains/wire records, complete message-tree ownership, replica ordering, source readability, deterministic conflicts/delete/restore/purge, compatibility/negotiation/byte limits, inert execution and twelve implementation acceptance cases. Eight atomic follow-ups were created through Backlog CLI; their design delivery map carries all original feature requirements. Living server-sync documentation distinguishes approval from activation. Self-review matched all nine amended design criteria and task dependencies; prose adds no executable feature tests. No source transport, schema migration, server write or runtime permission is enabled by closing this design task. Existing ADR121/124/008 and accepted ADR239 apply. Current code and real-server interoperability are qualified only by the follow-up tasks, never by this design record.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
