---
id: TASK-900
title: Add minimal disk-backed File Notes editor
status: To Do
assignee: []
created_date: '2026-07-27 14:32'
updated_date: '2026-07-27 14:37'
labels:
  - notes
  - library
  - files
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-07-27-minimal-file-notes-design.md
  - backlog/decisions/029-file-notes-disk-authority.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users manage one existing Git-backed Markdown/text folder from Library while disk and filename/path remain authoritative. Chatbook writes ordinary filesystem changes and keeps one separate SQLite replica for search and recovery without changing Database Notes or Git workflows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 User can choose and persist one notes root, browse its folder tree, and see a missing root as offline without filesystem or replica mutation.
- [ ] #2 Search replaces the tree and opens matching Markdown/text files.
- [ ] #3 Editor changes write directly to disk with debounced hash-checked saves, while unresolved dirty/conflict/error states prevent navigation from discarding the draft.
- [ ] #4 Valid leading frontmatter, BOM, uniform newline style, and final-newline state are preserved while the body is edited; unsafe text files remain path-visible and read-only.
- [ ] #5 Create, rename, move, delete, and restore operate on actual files beneath the root.
- [ ] #6 Delete commits a SQLite recovery snapshot and tombstone before unlinking.
- [ ] #7 SQLite is a separate root-namespaced replica with opt-in history for protected paths, and protected writes cannot occur before their pre-edit checkpoint commits.
- [ ] #8 Changed this session lists only Chatbook mutations.
- [ ] #9 Existing Database Notes and Git behavior remain unchanged.
- [ ] #10 External changes detected before publication are never silently overwritten, reconciliation does not remount an open editor, and scanning does not block the UI.
- [ ] #11 Restorable deletions remain discoverable after restart.
<!-- AC:END -->
