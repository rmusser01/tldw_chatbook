---
id: TASK-399.9
title: B2 Delete files with verified minimal restore
status: To Do
assignee: []
created_date: '2026-07-23 14:24'
labels:
  - notes
  - filesystem
  - recovery
dependencies:
  - TASK-399.8
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users delete an actual file only when Chatbook can guarantee its exact recovery through the smallest safe restore workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Delete confirmation displays the full selectable, wrapped, untruncated root label and relative path as literal text, states that disk is authoritative and the actual file will be removed, and names the guaranteed recovery expiry; its token binds the canonical root identity, path, UUID, expiry, and freshly verified raw hash, and a stale confirmation changes nothing.
- [ ] #2 Before mutation, a self-contained deletion revision and tombstone commit, round-trip verify, and retain exact bytes plus the supported metadata manifest for at least 30 days.
- [ ] #3 Quarantine preserves a late-changing target; only the confirmed exact file is unlinked and Git immediately sees the deletion.
- [ ] #4 Delete remains unavailable unless recovery health, capacity, supported-metadata round-trip, quarantine, and restore prerequisites pass; unsupported metadata keeps the file read-only.
- [ ] #5 Minimal restore reapplies and verifies exact bytes plus supported metadata only to an absent original path using no-replace publication and reuses the tombstoned UUID.
- [ ] #6 If the original path is occupied or its parent is missing, only exact export is offered; overwrite, alternate-path restore, and broad history remain absent.
- [ ] #7 Completion reports Deleted from disk with the guaranteed recovery expiry.
- [ ] #8 Fault tests cover every delete/minimal-restore journal, publication, projection, and completion boundary, including stale tokens, late changes, quarantine crashes, interrupted restore, full/corrupt recovery, expired payloads, and occupied/missing-parent refusal.
<!-- AC:END -->
