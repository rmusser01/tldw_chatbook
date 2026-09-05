---
id: TASK-31231
title: Round-trip Canvas history through Chatbook archives
status: In Progress
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-05 01:05'
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
- [ ] #1 Archives containing Canvas use Chatbook format 3.0 while archives without Canvas remain eligible for 2.0
- [ ] #2 Export includes inert manifests and exact source files with Canvas/revision ancestry, titles, provenance, runtime profiles, digests, sizes, and deletion metadata
- [ ] #3 Import validates paths, counts, declared and actual uncompressed sizes, UTF-8, digests, duplicate identities, cycles, parent ownership, and origin messages before mutation or rendering
- [ ] #4 Same-identity restore is digest-idempotent and refuses conflicting content without silent overwrite
- [ ] #5 Import-as-new remaps conversation, message, Canvas, revision, parent, origin, and reopen-hint identities as one graph
- [ ] #6 Unsupported runtime profiles remain inert and never execute under a guessed or weaker profile
- [ ] #7 V1/V2 archives retain existing behavior and Canvas data stays excluded from all synchronization paths
- [ ] #8 Export and import remain atomic under interruption or injected write failure
- [ ] #9 Focused unit, property, decompression-bomb, transaction, backward-compatibility, and whole-graph round-trip tests pass
<!-- AC:END -->

## Implementation Plan

ADR required: yes
ADR path: backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md
Reason: archive schema/versioning, graph identity remapping, atomic restore behavior, inert-source handling, and the local-only synchronization boundary are architectural decisions already accepted by ADR-115; this delivery implements and records the concrete format without creating a duplicate ADR.

1. Characterize the current Chatbook V1/V2 models, creator/importer transaction boundaries, limits, and compatibility fixtures before defining format 3.0.
2. Add typed Canvas archive records and inert deterministic paths, select V3 only when Canvas content is present, and document every field, limit, remapping rule, and unsupported-runtime behavior.
3. Export the complete bounded Canvas/revision/hint graph without compiling or rendering source, recomputing source byte counts and digests while streaming.
4. Validate the entire compressed archive and remapped graph before mutation, then implement digest-idempotent same-identity restore and import-as-new in one transaction.
5. Add failure-injection, decompression-bomb, property, graph, backward-compatibility, and synchronization-exclusion tests; run only targeted archive/repository suites.
6. Request independent code/security review, inspect a produced archive manually, update ADR-115 and this task with the final format/evidence, and mark Done only after every acceptance criterion passes.
