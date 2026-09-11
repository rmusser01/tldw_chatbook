---
id: TASK-32256
title: >-
  Library Notes Add from files: both relationship explanations are cut mid- word
  at 235 columns
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 15:23'
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
Residual of task-32125, which correctly moved the two relationships to sibling buttons under their own descriptions (C cap 26, D cap 21, with only Back to Notes pinned below). The descriptions themselves are truncated mid-word at 235x52, so the one screen whose entire job is to explain the difference between "Import once" and "Keep a folder synced" delivers neither explanation in full -- to a first-timer, at the point of an irreversible-feeling choice.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both relationship descriptions render complete at 235x52
- [ ] #2 Neither description is cut mid-word at any supported width: below the width where they fit they wrap or disclose
- [ ] #3 Covered by a test asserting the full description text at 235x52
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the mid-word cut at 235x52 against the cited captures and a rendered probe.\n2. If it reproduces, make the descriptions wrap; if not, pin the full text with a render-level test and report the non-reproduction.
<!-- SECTION:PLAN:END -->
