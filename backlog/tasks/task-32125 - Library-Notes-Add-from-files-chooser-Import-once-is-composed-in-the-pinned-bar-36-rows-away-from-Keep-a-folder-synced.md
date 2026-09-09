---
id: TASK-32125
title: >-
  Library Notes Add from files chooser: Import once is composed in the pinned bar 36 rows away from Keep a folder synced
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
  - layout
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PROVEN: `_compose_phase` (phase 'choose') yields only the keep-synced button under the two descriptions; `_compose_pinned_actions` yields 'Import once' beside 'Back to Notes' at the bottom of the canvas. A top-down reader sees a one-option choice, and the header already reads 'Lasting sync · Choose how files should relate…' before anything was chosen. Both assessors flagged it independently. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Import once and Keep a folder synced render as sibling buttons directly under their own descriptions, in description order
- [ ] #2 The pinned bar holds only Back to Notes on the choose phase
- [ ] #3 The header does not name a relationship before one is chosen
- [ ] #4 Covered by a compose test
<!-- AC:END -->
