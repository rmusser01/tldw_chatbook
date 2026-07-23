---
id: TASK-440
title: Roleplay readiness surfaces reflect character-chat provider readiness
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
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live: the workbench header showed "Ready" and the inspector "Console ready" while character replies were impossible (character provider unready - see task-425). Readiness badges currently reflect internal wiring, not "a character reply will work". They should incorporate the resolved character provider's readiness or say specifically what is and is not ready.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When the character-chat provider is not ready, Roleplay readiness surfaces say so (and what to do), rather than showing Ready
- [ ] #2 Inspector Console-readiness reflects the actual send path for the staged intent
<!-- AC:END -->
