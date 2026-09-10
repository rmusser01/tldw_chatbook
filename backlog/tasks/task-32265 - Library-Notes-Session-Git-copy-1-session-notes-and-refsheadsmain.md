---
id: TASK-32265
title: >-
  Library Notes Session Git copy: 1 session notes, and refs/heads/main
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - file-notes
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two copy defects on the flow that is otherwise the best-designed on the screen: "1 session notes will be committed" and "Committed 1 session notes as bd746be6..." do not pluralise, and the branch is displayed as `refs/heads/main` rather than `main`.

Both are on the pre-commit disclosure screen, which the design assessor called the most honest copy in the product; these are the only two places on it that read as machine output.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The session-note count pluralises correctly, including the one-note case
- [ ] #2 The branch renders as its short name
- [ ] #3 Covered by a test on the one-note case
<!-- AC:END -->
