---
id: TASK-399
title: >-
  File-backed Notes disk-authoritative Library management and local recovery
  replica
status: To Do
assignee: []
created_date: '2026-07-23 04:06'
updated_date: '2026-07-23 05:07'
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
Umbrella epic for the core phased delivery of first-class management for one existing Git-managed Markdown/text root in Library Notes. Disk remains authoritative and the user's Git workflow stays external; SQLite provides a derived local projection plus an independent same-device recovery replica for selected files and folders. Implementation planning must create atomic child tasks/PRs before code work begins. Optional database-pair backup, file RAG, recovery-only in-place restore, additional linked roots, recursive folder operations, and Git controls are separately gated follow-ups and do not block this epic's core completion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can preview and link one existing supported note root, browse its folder tree, and search deep and 5,000-sibling fixtures without Chatbook altering source bytes or replacing/degrading the current zero-root Database Notes UI/runtime aside from unobtrusive Link and detached-recovery management actions.
- [ ] #2 Filename and relative path remain canonical, opaque frontmatter is preserved byte-for-byte while the body is edited separately, and Database notes retain their existing behavior.
- [ ] #3 Selected files and folders have a self-contained current recovery replica and retained checkpoints in a separate notes_recovery.db; confirmed Chatbook deletions preserve round-trip-verified exact bytes for 30 days and can be restored safely.
- [ ] #4 Legacy folder sync, remote Sync v2, public database update/delete paths, and MCP cannot mutate file-backed rows or disk; global ownership gates reject overlapping roots and fence in-flight legacy mutation.
- [ ] #5 Library Notes provides the approved zero-root, linking, protection, tree/search/editor, capability/recovery-state, and wide/narrow UX; it preserves preview/export/template/Console handoff parity without adding Git controls.
- [ ] #6 Required cross-version exclusion, pair-migration, activation, kernel lease, path-containment, quota/corruption, rollback, durable commit/fsync, performance, crash-safety, watcher-storm, and zero-configuration non-degradation gates pass for the core rollout phases.
- [ ] #7 Users can create, edit, save, move, rename, create folders, remove empty folders, and confirm-delete supported files through versioned-hash operations that retain atomically displaced bytes; external changes reconcile near real time and unresolved drafts/conflicts remain durable and recoverable.
<!-- AC:END -->
