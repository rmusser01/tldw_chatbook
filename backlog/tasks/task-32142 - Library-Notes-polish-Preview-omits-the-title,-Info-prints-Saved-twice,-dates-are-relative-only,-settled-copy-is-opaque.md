---
id: TASK-32142
title: >-
  Library Notes polish: Preview omits the title, Info prints Saved twice, dates are relative only, settled copy is opaque
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Minor observations from both assessors: the Preview mode shows the body without the title; the Info view repeats 'Saved' in the header and above the panel; 'Created 3m · Modified now' is the only date anywhere for a note; 'All planned items settled.' ends the import; the delete receipt survived an entire import journey. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Preview shows the title above the rendered body
- [ ] #2 Saved appears once per view
- [ ] #3 Info shows an absolute timestamp beside the relative one
- [ ] #4 Receipts are dismissed when the user leaves the list for another workflow
<!-- AC:END -->
