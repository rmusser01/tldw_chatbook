---
id: TASK-438
title: Alternate greeting selector in Roleplay preview
status: In Progress
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-23 22:15'
labels:
  - roleplay
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). The card view shows "Alternate greetings: N" with their text, but the preview always seeds the primary first_mes; there is no way to start a session from an alternate greeting anywhere in the app.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When a character has alternate greetings, the preview offers a way to pick which greeting seeds the conversation
- [ ] #2 Reset returns to the chosen greeting, not silently to the primary one
<!-- AC:END -->
