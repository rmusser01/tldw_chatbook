---
id: TASK-32134
title: >-
  Library Notes Import once: a selected source cannot be changed, Add another file is absent for a folder, and Last import is never offered after Back to Notes
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evidence assessor: after a wrong selection the configure phase offers only Check selection and Back to Notes; the guide's 'Add another file' does not render for a folder selection. Both assessors: after a completed import and Back to Notes no 'Last import' control appears; `can_revisit_receipt` requires the RECEIPT phase or a SELECT phase with nothing selected, which Back to Notes does not leave behind (INFERRED). Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The configure phase offers Change selection and Clear
- [ ] #2 Last import is offered in the Notes list while a same-session receipt exists
- [ ] #3 The guide matches the shipped controls
<!-- AC:END -->
