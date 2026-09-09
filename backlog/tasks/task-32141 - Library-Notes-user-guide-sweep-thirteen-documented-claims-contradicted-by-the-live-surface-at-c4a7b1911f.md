---
id: TASK-32141
title: >-
  Library Notes user-guide sweep: thirteen documented claims contradicted by the live surface at c4a7b1911f
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - docs
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the critique's docs-vs-live table: no empty-state copy renders; rows show no age; rename does not update the row until you leave the field despite a 2026-09-06 'Verified against' stamp; the status line shows no word count; Undo/Dismiss not visible; 'Last import' never appears; the Sort strip has no Title; '/' also types; only Rename fits of Rename / Move / Remove; compact shows '‹ Notes'; 'Linked — folder' holds only for a clean first pick and the timeout copy never paints; 'Add another file' is not rendered for a folder selection. Some are fixed by sibling tasks; the rest need the prose corrected. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed claim is re-verified live or corrected in notes.md / file-notes.md
- [ ] #2 The 2026-09-06 rename-propagation stamp is corrected to describe the deferred-refresh behaviour shipped in #2531
- [ ] #3 Every 'Verified against' stamp on the two pages names the commit it was checked at
<!-- AC:END -->
