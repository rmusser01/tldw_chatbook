---
id: TASK-97
title: Resolve Notes sync conflicts inline
status: In Progress
assignee:
  - '@codex'
created_date: '2026-06-11 17:05'
updated_date: '2026-08-22 23:20'
labels: []
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-08-22-task-97-notes-sync-conflict-resolution-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the retired ASK-engine modal concept with reviewed inline conflict resolution in the lasting Database Notes sync flow. Users compare the current file and Library note, stage Keep file, Keep note, Keep both, or Skip for now without mutation, then apply an explicitly reviewed subset through the existing durable runtime and recovery journal. Conflict outcomes remain device-local, recoverable, and visible in bounded resolution history; deletion review remains outside this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A content conflict appears in the existing inline reviewed-sync flow with a bounded file/note comparison and Keep file, Keep note, Keep both, and Skip for now choices
- [ ] #2 Staging or changing a choice performs no mutation; selections survive paging but are discarded when the review becomes stale or is abandoned
- [ ] #3 Apply reviewed revalidates fresh authority and applies safe actions plus explicitly selected conflict resolutions; skipped conflicts remain Needs attention
- [ ] #4 Keep file and Keep note resolve only that occurrence without changing the root's configured direction
- [ ] #5 Keep both durably preserves the original note as an unbound manual conflict copy before updating the bound note, supports restart recovery, and records both outcomes
- [ ] #6 Durable resolution history records each explicit choice and offers per-item Undo only while exact recovery remains valid; later edits are never overwritten
- [ ] #7 Deletion, pause, managed-placement, capability, and activation attention remain blocked and unchanged
- [ ] #8 Comparison content and private authority never enter persistent diagnostics or logs, and unsupported filesystem writes remain fail-closed
<!-- AC:END -->
