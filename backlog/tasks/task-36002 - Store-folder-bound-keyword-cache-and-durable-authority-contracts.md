---
id: TASK-36002
title: Store folder-bound keyword cache and durable authority contracts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-21 14:30'
updated_date: '2026-10-21 14:36'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-10-20-workspace-project-keyword-index.md
  - backlog/decisions/240-workspace-project-context-and-folder-bound-index.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide opt-in derived project discovery with storage and revocation ownership separate from Library records.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fresh bindings remain Off and project text is scoped to an opaque complete binding/data/root identity without Library Media, Notes, sync or backup payload.
- [ ] #2 Manifest, chunks, FTS, frontier and revision publish atomically; rollback, duplicate delivery and stale scope/job publication preserve coherent results.
- [ ] #3 Authority witnesses remain monotonic across mutation and remove/recreate; cross-process fences and durable claims prevent stale publication and competing writers.
- [ ] #4 Canonical schema migration, exact storage admission, activation and finite owner retirement succeed without eager runtime imports or unknown cache files.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement K1 exactly from the keyword plan plus reconciled current-source preflight: frozen key/policy/scope/authority/page/search/live-reader contracts; schema-v1 per-binding kernel store with atomic chunk/FTS/frontier/manifest and durable job/CAS state; Workspace canonical authority witness/tombstone migration plus exact recovery catalog successor; exact excluded-cache ownership/admission/activation/retirement and inert recovery imports. Write intended RED first, then targeted store/migration/owner/fence tests; zero Library-policy or one-shot worker changes. ADR required: yes; ADR path: backlog/decisions/240-workspace-project-context-and-folder-bound-index.md; applies ADR101/102/126/174 and228-sqlite-store-kernel; no new ADR needed. Independent task review before next task.
<!-- SECTION:PLAN:END -->
