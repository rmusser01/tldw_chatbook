---
id: TASK-31231
title: Round-trip Canvas history through Chatbook archives
status: Done
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-05 02:11'
labels:
  - canvas
  - chatbooks
  - export
  - import
dependencies:
  - TASK-31227
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make durable Canvas documents and immutable revision graphs portable through local conversation/Chatbook export and import while keeping source inert and preserving older archive compatibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Archives containing Canvas use Chatbook format 3.0 while archives without Canvas remain eligible for 2.0
- [x] #2 Export includes inert manifests and exact source files with Canvas/revision ancestry, titles, provenance, runtime profiles, digests, sizes, and deletion metadata
- [x] #3 Import validates paths, counts, declared and actual uncompressed sizes, UTF-8, digests, duplicate identities, cycles, parent ownership, and origin messages before mutation or rendering
- [x] #4 Same-identity restore is digest-idempotent and refuses conflicting content without silent overwrite
- [x] #5 Import-as-new remaps conversation, message, Canvas, revision, parent, origin, and reopen-hint identities as one graph
- [x] #6 Unsupported runtime profiles remain inert and never execute under a guessed or weaker profile
- [x] #7 V1/V2 archives retain existing behavior and Canvas data stays excluded from all synchronization paths
- [x] #8 Export and import remain atomic under interruption or injected write failure
- [x] #9 Focused unit, property, decompression-bomb, transaction, backward-compatibility, and whole-graph round-trip tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md
Reason: archive schema/versioning, graph identity remapping, atomic restore behavior, inert-source handling, and the local-only synchronization boundary are architectural decisions already accepted by ADR-121; this delivery implements and records the concrete format without creating a duplicate ADR.

1. Characterize the current Chatbook V1/V2 models, creator/importer transaction boundaries, limits, and compatibility fixtures before defining format 3.0.
2. Add typed Canvas archive records and inert deterministic paths, select V3 only when Canvas content is present, and document every field, limit, remapping rule, and unsupported-runtime behavior.
3. Migrate Canvas revision runtime-profile storage from schema 66 to 67 so well-formed bounded unknown profiles can be retained inert, while execution remains restricted to explicitly supported profiles; verify genuine-v66 migration, rollback, and fresh-schema parity.
4. Export the complete bounded Canvas/revision/hint graph without compiling or rendering source, recomputing source byte counts and digests while streaming.
5. Validate the entire compressed archive and remapped graph before mutation, then implement digest-idempotent same-identity restore and import-as-new in one transaction.
6. Add failure-injection, decompression-bomb, property, graph, backward-compatibility, and synchronization-exclusion tests; run only targeted archive/repository suites.
7. Request independent code/security review, inspect a produced archive manually, update ADR-121 and this task with the final format/evidence, and mark Done only after every acceptance criterion passes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Defined typed, frozen, source-free Chatbook 3.0 Canvas manifest records in `f53ef11c74`. V3 is selected only when Canvas content exists; V1/V2 behavior is unchanged. Canonical source paths are inert `canvas/<canvas-id>/<revision-id>.html.txt`, deterministic ordering is enforced, and unknown well-formed runtime profiles remain labeled but non-executable.
- Implemented complete graph export/import in `d62281fc12`. Export streams repository source, recomputes UTF-8 byte counts and SHA-256 digests, validates origins against the exact staged conversation graph, and emits deterministic ZIP metadata. Import validates the physical container and every member/path/content/graph boundary before extraction or mutation.
- Same-identity restore revalidates the full graph under `BEGIN IMMEDIATE` and is exact-metadata/digest idempotent; any conflicting content, order, owner, or lineage aborts. Import-as-new remaps conversation, message, Canvas, revision, parent, origin, and reopen-hint identities before committing messages and Canvas rows atomically.
- Added schema 66→67 so bounded, well-formed future runtime-profile identifiers can be stored inert without weakening the compiler/renderer support allowlist. Genuine-v66 migration, rollback, fresh-schema parity, repository constraints, and synchronization exclusion are tested.
- Security review fixes covered deleted historical origins, shared-descriptor validation/extraction, physical-size and replacement TOCTOU, locked idempotence, export snapshot consistency, revision-order comparison, duplicate Canvas-owner ambiguity, path depth/size/prefix collisions, and special ZIP members. Final rereview concluded SHIP with no Critical or Important findings.
- Final targeted evidence: 157 format/compatibility tests, 290 Chatbook creator/importer/service/transaction tests, 38 repository/migration tests, and the whole-graph fixture passed. Ruff, Python compilation, and diff checks were clean; only the existing dependency and combined-session file-descriptor warnings appeared.
- Manual inspection of a produced archive showed two Canvas documents and four deterministic `.html.txt` revisions, including a sibling branch, title change, soft deletion, deleted historical origin, reopen hint, and inert `canvas-v9` profile. Manifest IDs, parents, origins, sizes, and digests were understandable without executing content, and no runnable `.html` member existed. The disposable archive/database were removed after inspection.
- ADR check: existing ADR-121 was updated with format 3.0 fields and limits, container defenses, atomic identity semantics, schema 67, inert unsupported-profile behavior, sync exclusion, and manual checkpoint evidence. No new ADR was needed.
<!-- SECTION:NOTES:END -->
