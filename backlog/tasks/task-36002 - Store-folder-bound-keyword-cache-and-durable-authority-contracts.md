---
id: TASK-36002
title: Store folder-bound keyword cache and durable authority contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-10-21 14:30'
updated_date: '2026-10-21 15:21'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-10-20-workspace-project-keyword-index.md
  - backlog/decisions/241-workspace-project-context-and-folder-bound-index.md
---


Current governance: [ADR-241](../decisions/241-workspace-project-context-and-folder-bound-index.md). Renumbered from project ADR-240 during 2026-10-22 dev integration; historical rulings and delivery notes below retain their original source-era labels.

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide opt-in derived project discovery with storage and revocation ownership separate from Library records.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh bindings remain Off and project text is scoped to an opaque complete binding/data/root identity without Library Media, Notes, sync or backup payload.
- [x] #2 Manifest, chunks, FTS, frontier and revision publish atomically; rollback, duplicate delivery and stale scope/job publication preserve coherent results.
- [x] #3 Authority witnesses remain monotonic across mutation and remove/recreate; cross-process fences and durable claims prevent stale publication and competing writers.
- [x] #4 Canonical schema migration, exact storage admission, activation and finite owner retirement succeed without eager runtime imports or unknown cache files.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement K1 exactly from the keyword plan plus reconciled current-source preflight: frozen key/policy/scope/authority/page/search/live-reader contracts; schema-v1 per-binding kernel store with atomic chunk/FTS/frontier/manifest and durable job/CAS state; Workspace canonical authority witness/tombstone migration plus exact recovery catalog successor; exact excluded-cache ownership/admission/activation/retirement and inert recovery imports. Write intended RED first, then targeted store/migration/owner/fence tests; zero Library-policy or one-shot worker changes. ADR required: yes; ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md; applies ADR101/102/126/174 and228-sqlite-store-kernel; no new ADR needed. Independent task review before next task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented schema-v1 folder-bound kernel/FTS cache with immutable authority/page/search contracts, prepared chunk offsets, atomic manifest/frontier/FTS receipts, exact durable job claims and retained epoch/CAS history. Workspace canonical schema-v9 authority witnesses and frozen recovery successor preserve cross-process mutation serialization and tombstones; exact private owner/admission/activation/finite and maintenance retirement are registered, cache bytes intentionally excluded from backup. No Library or semantic behavior changed. ADR240 applies (ADR101/102/126/174/228); no new ADR. Source commit8bdb4be36df66222dc5f13ed6257ff86eb539b14. Independent spec/quality review approved K1; minor adapter return annotation deferred to final review. Final amended targeted selection25passed1inheritedwarning46.74s; broader selection183passed10platformskips4failed422.43s, with both K1 failures corrected and two untouched census failures reproduced on exact33fa199 export2failed148.23s. New-file Ruff/format and whitespace pass;25existing Ruff findings unchanged, no new findings. No full suite or other-platform claim. Evidence and integration gates: Docs/superpowers/qa/2026-10-20-workspace-project-context/keyword/execution-notes.md; managed task-1 report/review. K3 must supply actual profile/restore authority issuance, physical worker retirement and fenced publication; K4 live disclosure and K2 bounded scanning remain their separate scopes.
<!-- SECTION:NOTES:END -->
