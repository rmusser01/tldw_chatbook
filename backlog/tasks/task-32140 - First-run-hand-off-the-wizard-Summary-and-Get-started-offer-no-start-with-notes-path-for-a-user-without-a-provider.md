---
id: TASK-32140
title: >-
  First-run hand-off: the wizard Summary and Get started offer no start-with-notes path for a user without a provider
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - onboarding
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both assessors on a fresh profile: the Summary step's actions are provider setup, Explore Home and settings; the post-setup Console card lists provider, model and first message. A local-first user who came for notes is told the only thing they can do needs an API key. task-32072 (merged in #2531) added 'Add your first document'; a notes path is still missing. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Summary step and the Get started card offer 'Write your first note' that lands in Library ▸ Notes ▸ New note
- [ ] #2 Covered by a wizard test
<!-- AC:END -->
