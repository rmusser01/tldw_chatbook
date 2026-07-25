---
id: TASK-442
title: Active persona concept with user-name substitution in chats
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
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Personas (who you are) are currently one-shot staged text for Console; there is no default/active persona, no import, and the preview always renders the user as "you"/"User" (placeholders replace {{user}} with the literal "User"). An RP user expects to pick a persona once and have their name/description flow into greetings, placeholder substitution, and sends.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A persona can be marked active/default and persists across sessions
- [ ] #2 Preview and character sends substitute the active persona's name for {{user}} and label the user's messages with it
- [ ] #3 With no active persona, current behavior is unchanged
<!-- AC:END -->
