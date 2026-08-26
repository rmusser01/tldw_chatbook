---
id: TASK-22301
title: Bound Library rail width across modes
status: To Do
assignee: []
created_date: '2026-08-26 03:31'
labels:
  - library
  - ux
  - layout
dependencies: []
priority: high
references:
  - Docs/superpowers/specs/2026-08-25-library-rail-bounded-width-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the persistent Library navigation rail visually stable across every Library mode by retaining fractional sizing while bounding it around the approved Collections reference width.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The persistent Library rail keeps `3fr` sizing with an exact 24-cell minimum and 34-cell maximum in every Library mode.
- [ ] #2 Switching among Media, Chats, Notes, Prompts, Skills, Collections, Search / RAG, Import, and Export does not change the rail sizing contract.
- [ ] #3 At supported wide and standard terminal widths, the rendered rail stays within 24–34 cells while the canvas remains contained and receives the remaining width.
- [ ] #4 Compact terminal layouts preserve the 24-cell readable rail floor without introducing new canvas or footer overlap.
- [ ] #5 Production-styled tests cover initial mount, mode switching, live resize, and 235-, 170-, 120-, 100-, 80-, and 60-column geometry.
- [ ] #6 Library documentation records the bounded fractional rail behavior and current limits.
<!-- AC:END -->
