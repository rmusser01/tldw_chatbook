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
- [ ] #1 User can select one notes root and browse its folder tree.
- [ ] #2 Search replaces the tree and opens matching Markdown/text files.
- [ ] #3 Editor changes write directly to disk with debounced hash-checked saves.
- [ ] #4 Valid leading frontmatter is preserved byte-for-byte while the body is edited.
- [ ] #5 Create, rename, move, delete, and restore operate on actual files beneath the root.
- [ ] #6 Delete commits a SQLite recovery snapshot and tombstone before unlinking.
- [ ] #7 SQLite is a separate replica with opt-in history for protected paths.
- [ ] #8 Changed this session lists only Chatbook mutations.
- [ ] #9 Existing Database Notes and Git behavior remain unchanged.
- [ ] #10 External changes detected before publication are never silently overwritten.
<!-- AC:END -->
