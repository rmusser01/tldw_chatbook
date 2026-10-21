---
id: TASK-36003
title: Scan project folders in bounded resumable pages
status: To Do
assignee:
  - '@codex'
created_date: '2026-10-21 14:30'
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
