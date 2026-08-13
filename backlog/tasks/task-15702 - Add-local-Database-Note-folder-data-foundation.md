---
id: TASK-15702
title: Add local Database Note folder data foundation
status: In Progress
assignee: []
created_date: '2026-08-13 01:33'
labels:
  - notes
  - folders
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-12-notes-folder-import-sync-design.md
  - backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md
  - >-
    backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the durable local Database Note folder schema, ownership-aware memberships, repository, and normalized service contract required by the later navigator, one-time import, and lasting-sync slices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fresh and supported prior ChaChaNotes schemas migrate atomically to the folder schema without inventing memberships for existing notes.
- [ ] #2 Users can create, rename, move, soft-delete, and restore nested folders with optimistic version checks and case/Unicode collision protection.
- [ ] #3 A note can hold multiple manual folder memberships plus owner-scoped managed membership, and removing organization never deletes the note.
- [ ] #4 The normalized folder service exposes typed local folder, membership, capability, and paging operations while unsupported remote mutation remains explicit.
- [ ] #5 Folder and membership reads are bounded and bulk-loaded without per-note queries.
- [ ] #6 ChaChaNotes backup and restore preserve folder organization, and managed memberships without a restored device owner remain inactive for review.
- [ ] #7 Focused migration, repository, service, backup, performance, and contract tests cover the new behavior without changing flat note editing or Sync-v2 M1 payloads.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the ChaChaNotes v36 folder and ownership-aware membership schema plus atomic migration coverage for fresh and prior databases.
2. Implement a focused local folder repository with normalized paths, optimistic subtree mutations, bulk reads, owner-scoped memberships, and inactive-owner restore review.
3. Add typed folder models and route folder operations through the existing Notes scope service without changing Sync-v2 M1 payloads.
4. Verify backup behavior, performance/query bounds, migrations, and local/remote capability contracts.

ADR required: yes

ADR paths:

- `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`
- `backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md`

Reason: this slice implements the accepted local folder schema, ownership,
backup, normalized service, and bounded paging boundaries.

Detailed executable plan:
`Docs/superpowers/plans/2026-08-12-local-database-note-folder-foundation.md`
<!-- SECTION:PLAN:END -->

## Definition of Done

- [ ] Every acceptance criterion is checked with automated or recorded evidence.
- [ ] Focused tests, broader DB/Notes regressions, and static analysis pass.
- [ ] Implementation Notes summarize the approach, files, trade-offs, and evidence.
- [ ] ADR-059 and ADR-060 are linked from implementation notes and remain satisfied.
- [ ] A self-review finds no raw folder SQL outside the repository or Sync-v2 payload drift.
- [ ] The task is set to Done only after all requirements above are complete.
