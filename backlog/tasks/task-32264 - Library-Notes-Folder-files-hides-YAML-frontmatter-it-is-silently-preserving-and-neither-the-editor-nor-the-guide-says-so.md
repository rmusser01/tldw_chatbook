---
id: TASK-32264
title: >-
  Library Notes Folder files hides YAML frontmatter it is silently preserving,
  and neither the editor nor the guide says so
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - file-notes
  - docs
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified both ways by both assessors: the editor does not display a file's YAML frontmatter, and an in-app edit saved with the frontmatter intact on disk. So the app is hiding text it is faithfully preserving, and `Docs/User_Guide/library/file-notes.md` never mentions either half.

A user editing a vault file in Chatbook has no way to know whether their Obsidian properties survived the round trip -- and the answer is that they did.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The editor indicates that hidden frontmatter exists on a file that has it, and that it is preserved
- [ ] #2 `Docs/User_Guide/library/file-notes.md` documents the behaviour
- [ ] #3 Covered by a test asserting the indication for a file with frontmatter
<!-- AC:END -->
