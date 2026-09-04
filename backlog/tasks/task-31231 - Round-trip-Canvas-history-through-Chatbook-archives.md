---
id: TASK-31231
title: Round-trip Canvas history through Chatbook archives
status: To Do
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-03'
labels: [canvas, chatbooks, export, import]
dependencies: [TASK-31227]
priority: medium
---

## Description

Make durable Canvas documents and immutable revision graphs portable through local conversation/Chatbook export and import while keeping source inert and preserving older archive compatibility.

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

## Related Design

- `Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md`
- `Docs/superpowers/plans/2026-09-03-chatbook-canvas-implementation.md`
- `backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md`
