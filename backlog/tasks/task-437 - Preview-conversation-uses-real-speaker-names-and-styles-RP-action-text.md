---
id: TASK-437
title: Preview conversation uses real speaker names and styles RP action text
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - roleplay
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live: preview transcript labels speakers literally as "character:" and "you:" instead of the card name and persona/user name, and RP *action* asterisks render as raw text. Small changes with outsized effect on how in-genre the surface feels.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Preview messages are labelled with the character's actual name (and the persona/user name once available)
- [ ] #2 Single-asterisk action/emphasis spans render styled rather than as literal asterisks
<!-- AC:END -->
