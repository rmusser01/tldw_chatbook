---
id: TASK-36003
title: Scan project folders in bounded resumable pages
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-21 14:30'
updated_date: '2026-10-21 15:22'
labels: []
dependencies:
  - TASK-36002
documentation:
  - Docs/superpowers/plans/2026-10-20-workspace-project-keyword-index.md
  - backlog/decisions/240-workspace-project-context-and-folder-bound-index.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Discover large local project trees without unbounded memory, stale authority or leaked excluded content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pages stay within256metadata entries, two reads,4MiB queued text and closed bounded pipe frames; eligible UTF8 files are at most1MiB.
- [ ] #2 Sensitive/excluded, binary/generated and symlink/junction/reparse content is skipped before descent/read; changed root or revoked unit authority stops new reads.
- [ ] #3 Incomplete directories and interruptions remain truthful and resumable without duplicate publication or false deletion proof.
- [ ] #4 Real worker retirement and20k wide/many-directory corpus receipts demonstrate resource bounds and report actual platform qualification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement approved K2 bounded stdlib-only scanner and closed protocol with real contained-process/root-pin transport, per-unit authority admission and narrowing exclusion floor, same-read capture helper and private Capture operation, durable completion receipts and bounded backpressure. Consume K1 immutable records; emit raw ScanPage and leave chunk preparation/publication/claims to K3. TDD first, then actual worker/root-pin and 20k-file/directory/wide-corpus qualification with resource/cancellation/retirement receipts, scoped lint and independent review. ADR required: yes, existing ADR240-workspace-project-context-and-folder-bound-index applies with ADR101/102/174/181; no new ADR. Preserve Library choices; no semantic/index UI/service or ordinary one-shot protocol changes.
<!-- SECTION:PLAN:END -->
