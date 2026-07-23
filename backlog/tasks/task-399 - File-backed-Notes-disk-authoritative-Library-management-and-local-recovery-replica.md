---
id: TASK-399
title: >-
  File-backed Notes disk-authoritative Library management and local recovery
  replica
status: To Do
assignee: []
created_date: '2026-07-23 04:06'
updated_date: '2026-07-23 04:25'
labels:
  - notes
  - library
  - storage
  - recovery
  - epic
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Umbrella epic for phased delivery of first-class management for existing Git-managed Markdown and text folders in Library Notes. Disk remains authoritative and the user's Git workflow stays external; SQLite provides a derived local projection plus an independent same-device recovery replica for selected files and folders. Implementation planning must create atomic child tasks/PRs before code work begins.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can link existing supported note roots, browse their folder trees, and search at 5,000-file scale without Chatbook altering source bytes or starting file services when no root is linked.
- [ ] #2 Filename and relative path remain canonical, opaque frontmatter is preserved byte-for-byte while the body is edited separately, and Database notes retain their existing behavior.
- [ ] #3 Selected files and folders have a self-contained current recovery replica and retained checkpoints in a separate notes_recovery.db; confirmed Chatbook deletions preserve exact verified bytes for 30 days and can be restored safely.
- [ ] #4 Legacy folder sync, remote Sync v2, public database update/delete paths, and MCP cannot mutate file-backed rows or disk; overlapping ownership is rejected.
- [ ] #5 Library Notes provides the approved tree/search/editor UX on wide and narrow terminals, preserves existing preview/export/template/Console handoff parity, and does not add Git stage/commit/push controls in this tranche.
- [ ] #6 Migration, activation, lease, quota, corruption, backup, rollback, performance, crash-safety, watcher-storm, and zero-configuration non-degradation gates pass for the approved rollout phases.
- [ ] #7 Users can create, edit, save, move, rename, and confirm-delete supported files through hash-checked operations; external changes reconcile near real time and unresolved drafts or conflicts remain durable and recoverable.
<!-- AC:END -->
