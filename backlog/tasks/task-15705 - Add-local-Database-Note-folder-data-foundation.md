---
id: TASK-15705
title: Add local Database Note folder data foundation
status: Done
assignee: []
created_date: '2026-08-13 01:33'
updated_date: '2026-08-13 06:15'
labels:
  - notes
  - folders
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-12-notes-folder-import-sync-design.md
  - backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md
  - >-
    backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the durable local Database Note folder schema, ownership-aware memberships, repository, and normalized service contract required by the later navigator, one-time import, and lasting-sync slices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh and supported prior ChaChaNotes schemas migrate atomically to the folder schema without inventing memberships for existing notes.
- [x] #2 Users can create, rename, move, soft-delete, and restore nested folders with optimistic version checks and case/Unicode collision protection.
- [x] #3 A note can hold multiple manual folder memberships plus owner-scoped managed membership, and removing organization never deletes the note.
- [x] #4 The normalized folder service exposes typed local folder, membership, capability, and paging operations while unsupported remote mutation remains explicit.
- [x] #5 Folder and membership reads are bounded and bulk-loaded without per-note queries.
- [x] #6 ChaChaNotes backup and restore preserve folder organization, and managed memberships without a restored device owner remain inactive for review.
- [x] #7 Focused migration, repository, service, backup, performance, and contract tests cover the new behavior without changing flat note editing or Sync-v2 M1 payloads.
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
- `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`

Reason: this slice implements the accepted local folder schema, ownership,
backup, normalized service, and bounded paging boundaries.

Detailed executable plan:
`Docs/superpowers/plans/2026-08-12-local-database-note-folder-foundation.md`
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added atomic ChaChaNotes v36 folder and ownership-aware membership storage,
  normalized frozen contracts, a repository with optimistic subtree mutations and
  bounded bulk reads, Notes scope routing, and app composition for local Database
  Notes. Folder deletion and membership removal never mutate note rows.
- Preserved logical folders in normal ChaChaNotes backup/restore. Restored managed
  memberships are inactive until their device owner is reviewed; manual and managed
  memberships remain independently removable. Unsupported server/workspace folder
  mutation is reported explicitly.
- Kept Sync-v2 M1 unchanged and added a recursive key guard proving folder,
  membership, owner, and binding fields do not enter note envelopes. Folder SQL is
  confined to the v36 migration and folder repository; the models/repository have
  no Textual dependency or device-private root-path storage.
- Representative measurement used 5,000 active notes, 500 folders, and 10,000
  memberships. Root load returned 500 folders in 0.005001s with 3 SELECTs;
  one expanded 500-note load took 0.003271s with 3 SELECTs. Tests also pin constant
  query shape at 10 and 500 placements and chunk large identifier sets. Tree reads
  now expose independent bounded folder, note, and membership cursors, cap consumed
  expanded-folder IDs, and stay below SQLite's supported 999-variable ceiling.
- Added every file-backed ChaChaNotes runtime migration through v36 to wheel and
  source-distribution metadata. Packaging tests inspect both artifacts and install
  the wheel into an isolated target, where a real v35 database upgrades to v36.
- Verification: folder-focused Ruff passed; the complete requested lint target has
  the same 873 pre-existing findings at base `8e12ded9b` and on this branch. Focused
  tests passed (289), two packaging regressions passed, and 92 affected legacy
  migration tests passed. A fresh broad run excluding the exact 13 failures already
  reproduced on the pre-feature checkout produced 2,445 passes and 48 skips. Those
  baseline failures are an optional NumPy check, a pre-existing schema allowlist
  check, and hardened Git tests incompatible with this Homebrew/sandbox filesystem.
  No folder-related failure remained. Independent final review approved the package,
  paging, compatibility, and query-bound fixes with no remaining issue.
- Repaired schema-version expectations and migration fixtures exposed by the v36
  bump, added the two folder tables to the SQL validation allowlist, and recorded the
  general fixture lesson in `backlog/docs/lessons-testing-evidence.md`.
- Post-rebase review hardened exact-folder reads and uniqueness handling around the
  shared transaction boundary, routed the isolated installed-wheel migration probe
  through central path validation, and completed Google-style API documentation.
  A same-millisecond tombstone regression verifies collision advancement, with
  explicit `Z`-to-UTC-offset normalization matching the repository convention.
- Architecture remains governed by
  `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md` and
  `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`;
  no new ADR was required because this slice directly implements those accepted
  boundaries.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Every acceptance criterion is checked with automated or recorded evidence.
- [x] #2 Focused tests, broader DB/Notes regressions, and static analysis pass or have counterfactual baseline evidence.
- [x] #3 Implementation Notes summarize the approach, files, trade-offs, and evidence.
- [x] #4 ADR-059 and ADR-073 are linked from implementation notes and remain satisfied.
- [x] #5 A self-review finds no raw folder SQL outside the repository or Sync-v2 payload drift.
- [x] #6 The task is set to Done only after all requirements above are complete.
<!-- DOD:END -->
