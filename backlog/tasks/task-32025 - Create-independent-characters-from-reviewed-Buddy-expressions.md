---
id: TASK-32025
title: Create independent characters from reviewed Buddy expressions
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 04:28'
updated_date: '2026-09-08 04:59'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users reuse a saved Buddy or native Buddy archive as an independent editable Console character with faithful reactions and preserved source attribution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native and saved Buddy sources are validated snapshots without dummy Personas; native expression import rejects corrupt archives using native limits.
- [x] #2 Reviewed mappings preserve all supported sequences and require collision resolution; faithful animated output or explicit static fallback preserves pixels and timing under resource limits.
- [x] #3 Creation atomically publishes an independent character without changing current chat or Buddy preferences; stale sources, cancellation and injected failures create no partial actor.
- [x] #4 Conversion lineage and artwork attribution survive editing, export and reimport; replacements remove stale lineage.
- [x] #5 Users can review, preview and create from local Buddy detail or native archive, then explicitly open the new character in Console.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement Docs/superpowers/plans/2026-09-07-buddy-character-conversion.md with bounded source/UI/lineage ownership and integrated conversion review. ADR required: yes; ADR-074 amended for one-time reviewed snapshots and lineage carrier. Validate source snapshots, encode and verify timelines, preserve portable lineage, publish atomically through Actor Packs, mount review and explicit Console open, then run targeted integration and self-review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented reviewed independent character creation from saved local Buddies and native archives, including direct Characters Import. Added immutable native snapshots, bounded faithful WebP/PNG conversion, independent portraits, editable mapping review, atomic Actor Pack publication and explicit Console navigation. Preserved public artwork terms and asset-bound lineage with carrier v2; replacements clear stale lineage. ADR-074 governs the boundary. Targeted evidence: 82 integration tests, 101 regressions, 51 UI/import/CSS tests and 20 final UI recovery tests passed (overlapping runs); final converter 15 and snapshot 16 passed. All seven collection archives converted and published in a disposable profile with no animation fallback or pending cleanup. New files pass lint/format and existing-file lint comparison adds no diagnostics. User guide, reviewed screenshots and verification record updated. Server lineage support remains unsupported; no server changes or live-server preservation claim. See Docs/superpowers/reviews/2026-09-07-buddy-character-conversion-verification.md.
<!-- SECTION:NOTES:END -->
