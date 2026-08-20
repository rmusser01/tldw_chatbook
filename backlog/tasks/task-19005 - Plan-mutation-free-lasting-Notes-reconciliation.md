---
id: TASK-19005
title: Plan mutation-free lasting Notes reconciliation
status: To Do
assignee: []
created_date: '2026-08-20 07:42'
labels:
  - notes
  - sync
  - filesystem
dependencies:
  - TASK-19004
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Classify local folder and Library observations into safe operations, attention items, skips, identity-proven moves, and deletion candidates without mutating either authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root admission and observation reject unsafe overlap, symlink traversal, aliases, hard links, unsupported encodings, unstable identities, and representation or metadata loss.
- [ ] #2 Supported UTF-8 BOM, newline, final-newline, mode, and identity profiles round-trip without silent normalization.
- [ ] #3 A pure deterministic planner applies the ADR direction matrix, classifies identity-proven moves before missing-side deletion candidates, and never chooses a conflict or deletion winner.
- [ ] #4 Out-of-direction changes, both-side changes, filesystem moves implied by note changes, ambiguous identity, and capability loss become explicit attention or skip actions.
- [ ] #5 Repeated planning is idempotent, creates no root, binding, receipt, recovery, note, folder, file, or configuration mutation, and rejects stale reviewed observations before apply.
- [ ] #6 Deletion-burst grouping uses measured representative-tree evidence rather than a speculative production threshold.
<!-- AC:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md`, `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: ADR-059/073 already define admission, representation preservation, direction, move ordering, and explicit-attention policy; low-level path primitives may be reused without File Notes authority.
