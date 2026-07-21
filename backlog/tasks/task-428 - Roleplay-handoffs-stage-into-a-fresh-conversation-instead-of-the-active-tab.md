---
id: TASK-428
title: Roleplay handoffs stage into a fresh conversation instead of the active tab
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - roleplay
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live: with an existing Console conversation open, Start Chat staged the character handoff into that same (unrelated, already polluted) tab, and the conversation ends up named after the prefilled meta-instruction ("Continue this con..."). Handoffs from the Roleplay workbench should always land in a fresh conversation/tab so prior context does not bleed into the character chat and vice versa.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Start Chat / Open in Console with another conversation active creates and focuses a new conversation rather than reusing the active tab
- [ ] #2 The new conversation is not named after prefilled instruction text
<!-- AC:END -->
