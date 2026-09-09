---
id: TASK-32130
title: >-
  Library Notes Import once copy: config and empty files reported as Failed, CSV fan-out undercounted in the review, receipt lists no skipped files
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
  - copy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed by the evidence assessor on the 71-file vault: `.obsidian/*.json` and an empty `Untitled.md` both read 'This source could not be imported safely.' (a config file is not a failed import and an empty file is not unsafe); `notes.csv` is reviewed as 'Content: create 1 new note' and imports two; the receipt '61 imported · 0 updated · 11 skipped · 0 failed' names none of the skipped paths or reasons, and 'All planned items settled.' is opaque. Independent of task-32129: these are honest-copy problems for any import. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Empty and whitespace-only files are reported as 'Empty, nothing to import' and config-like files as skipped with the reason
- [ ] #2 The review's per-item plan states the number of notes a structured file will create
- [ ] #3 The receipt has a 'Skipped (N)' disclosure listing each path and reason
- [ ] #4 The completion copy says what happened in plain words
<!-- AC:END -->
