---
id: TASK-32031
title: Import reviewed Petdex companions as native Buddies
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 05:09'
updated_date: '2026-09-10 15:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users import a public Petdex pet or downloaded package as a local Buddy, preserve creator and terms, review operational mappings, and reuse ordinary Buddy-to-character conversion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Public object and compact registries and local folder or ZIP sources resolve exact bounded immutable data; malformed or ambiguous inputs cannot mutate profiles.
- [x] #2 Remote fetch pins validated public IPs to HTTPS connections with correct hostname verification, bounded decoding and redirect validation; no proxy or private-target bypass.
- [x] #3 Classic and declared or manually reviewed v2 atlases preserve regions, counts and exact loop timing, with disclosed fallbacks and no guessed extra rows.
- [x] #4 Review imports only an unpublished native draft into the captured local Persona; cancellation, stale authority and failures preserve existing packs and clean owned staging.
- [x] #5 Creator, source and terms survive saving, offline playback, native export and reimport and independent character conversion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-09-07-petdex-import.md with independent transport, native artwork, source/conversion and UI ownership. ADR required: yes; ADR-145 defines strict pinned HTTPS, source mapping and preserved native terms. Validate immutable sources, map regions and timing, review into existing draft publication, then test native export/reimport and independent character conversion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented reviewed Petdex URL/slug and folder/ZIP imports into unpublished native Buddy drafts, with pinned public HTTPS, exact atlas mappings, static preview fallback, and guarded atomic native export. Original creator/source/terms survive editing, saving, offline export/reimport and independent animated character publication. ADR-145 governs the boundary; no migration or server mutation. Combined targeted regression run: 782 passed; final Petdex suite: 103 passed, including four additional archive/inventory cases. New files pass Ruff lint/format, existing edits add no diagnostics, CSS sync passes. Live public fetch and offline publication verified in a disposable profile. See Docs/superpowers/reviews/2026-09-07-petdex-import-verification.md and Docs/User_Guide/buddy.md.

Generalizable ZIP raw-name and Pillow decoded-image cleanup incidents are recorded in backlog/docs/lessons-bounded-artwork-imports.md.

2026-09-10 integration onto dev 16c72b5b1e: prior validation above is historical source-branch evidence; combined-code targeted qualification is in progress.

2026-09-10 integration complete; Petdex ADR renumbered to 145 after scanning cached refs. Final affected Petdex/UI/artwork/conversion run: 206 passed after restoring a public constant removed during validator consolidation. CSS reproducibility and no-new-diagnostics checks pass. No downloaded assets added. Independent Buddy management Petdex entry point and live installation qualification remain explicit follow-up scope; exact seams and fresh evidence: Docs/superpowers/reviews/2026-09-10-buddy-feature-integration-verification.md. Earlier notes are historical source-branch evidence.
<!-- SECTION:NOTES:END -->
