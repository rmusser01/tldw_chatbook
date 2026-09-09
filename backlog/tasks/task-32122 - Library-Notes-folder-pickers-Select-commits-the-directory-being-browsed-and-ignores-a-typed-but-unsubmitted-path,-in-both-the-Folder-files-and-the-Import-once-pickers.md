---
id: TASK-32122
title: >-
  Library Notes folder pickers: Select commits the directory being browsed and ignores a typed but unsubmitted path, in both the Folder files and the Import once pickers
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - file-notes
  - import
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PROVEN in code and live: `EnhancedSelectDirectory._select_viewed_directory` returns `_dir_nav().location`; the typed path only takes effect on Enter. The Import once picker is `FileOpen(offer_select_folder=True)`, whose field is labelled 'File name' and is never read by 'Select folder'. Both assessors typed the vault path and pressed Select and got the source repository (Import once) or the home directory (Folder files, which then triggered task-32121). The confirmation shows only a basename ('1 folder selected: notes-review'), so the mistake is invisible, and the picker's own Enter-vs-Select hint never appeared on screen. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pressing Select (or Select folder) after typing an absolute path selects that path, or shows an inline error naming why it did not resolve; it never silently substitutes the browsed directory
- [ ] #2 In folder mode the field is labelled 'Folder path', never 'File name'
- [ ] #3 The selection confirmation shows the absolute path, not the basename
- [ ] #4 Both pickers share one component, one start location rule (last used, else home) and one hint line that is visible at 235x52 and 100x30
- [ ] #5 Covered by tests for typed-path-then-Select in both pickers
<!-- AC:END -->
