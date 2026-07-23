---
id: TASK-399.8
title: B1b Add move conflict resolution and startup recovery
status: To Do
assignee: []
created_date: '2026-07-23 14:24'
labels:
  - notes
  - filesystem
  - recovery
dependencies:
  - TASK-399.7
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete safe writable editing with identity-preserving relocation, durable conflict resolution, and deterministic recovery from interrupted rename/move operations through the B1a classifier.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rename and move preserve UUID only after verified no-replace movement within the same writable root and an existing destination directory.
- [ ] #2 Existing, normalized-colliding, cross-device, unsafe, or unsupported destinations are rejected without changing either path.
- [ ] #3 Dirty-buffer external edits, moves, deletion, and late publication races durably retain base, draft, and latest disk sides before navigation releases the buffer.
- [ ] #4 Compare, save-as-new, keep-editing, overwrite, and discard re-hash disk and cannot erase the only copy of any side.
- [ ] #5 Startup extends the B1a classifier to interrupted rename/move using both paths, hashes, metadata fingerprints, and owned artifacts, and never blindly replays a filesystem mutation.
- [ ] #6 Shared-classifier regressions prove create/save and rename/move artifacts are removed only after exact ownership plus durable byte/metadata capture.
- [ ] #7 Root-offline, recovery/projection-unavailable, and mode-transition states remain actionable while Database Notes stays usable.
- [ ] #8 Crash tests at every journal/publication/projection/completion boundary preserve canonical disk bytes or a verified draft plus unambiguous Attention.
- [ ] #9 Folder creation, rename, move, and recursive mutation remain unavailable.
- [ ] #10 Offline/read-only transitions and disk-change conflicts never hide draft durability state or bypass its resolution/navigation guard.
- [ ] #11 Writable Unlink crosses the operation barrier and releases both leases while retaining drafts/recovery; Forget is blocked by pending/Attention/unresolved drafts and explicitly purges projection/FTS only after safe resolution.
- [ ] #12 The B1 release gate exposes create/save/autosave/rename/move only after the full conflict actions, shared startup classifier, and writable Unlink/Forget barriers pass together.
<!-- AC:END -->
