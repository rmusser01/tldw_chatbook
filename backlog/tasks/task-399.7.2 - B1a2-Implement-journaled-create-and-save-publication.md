---
id: TASK-399.7.2
title: B1a2 Implement journaled create and save publication
status: To Do
assignee: []
created_date: '2026-07-23 15:35'
labels:
  - notes
  - filesystem
  - recovery
dependencies:
  - TASK-399.7.1
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399.7
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the native path-safe mutation core for blank create and exact body save while preserving every byte and supported security fact that a Chatbook write could displace.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every command re-resolves a pinned root handle and no-follow relative path, rejects unsafe traversal, symlink, mount, identity, or existing-destination surprises, and admits only existing destination directories within the verified writable root.
- [ ] #2 Blank create and body save re-hash current disk state, durably record intent and expected state, verify mandatory safety bytes plus metadata, publish with the B0-proven no-replace or atomic replacement primitive, and mark completion only after disk, recovery, and projection agree.
- [ ] #3 Opaque frontmatter, BOM, uniform newline and final-newline state, and supported security metadata manifests and fingerprints round-trip exactly; mixed or lone-CR normalization requires hash-bound acknowledgment and verified prior bytes.
- [ ] #4 Every displaced side and any draft that could otherwise become the only copy is retained durably before publication; a conflict never overwrites observed disk bytes.
- [ ] #5 Projection publication precedes journal completion, while later FTS failure remains retryable, reports search-index-updating, and does not invalidate a completed source mutation.
- [ ] #6 No-op saves, stale editor generations, duplicate command submission, recovery-capacity refusal, permission changes, and crashes at every intent, safety-copy, publication, projection, and completion boundary are deterministic and fail closed.
- [ ] #7 This child exposes no autosave integration, rename, move, delete, folder mutation, writable control, or read/write transition.
<!-- AC:END -->
