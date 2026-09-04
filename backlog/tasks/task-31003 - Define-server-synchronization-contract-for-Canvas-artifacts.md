---
id: TASK-31003
title: Define server synchronization contract for Canvas artifacts
status: To Do
assignee: []
created_date: '2026-09-03 13:39'
updated_date: '2026-09-03 13:39'
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
