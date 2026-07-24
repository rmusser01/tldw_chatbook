---
id: TASK-442
title: Active persona concept with user-name substitution in chats
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-24 14:32'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DROPPED (mis-specified) 2026-07-24. The task inverts this app's macro semantics: {{user}} = the application USER (the human); {{persona}}/{{char}}/{{character}} = the CHARACTER/persona (AI side). AC#2 ('substitute the active persona's name for {{user}}') would write the character-side name INTO the human placeholder, corrupting {{user}}. The whole 'Personas = who you are' premise this task is built on is wrong per the app author. replace_placeholders (Character_Chat_Lib.py:404) already maps {{user}}->user and {{char}}->char correctly (it lacks {{character}}/{{persona}} aliases). If a real gap is pursued later, re-file it correctly (e.g. add {{character}}/{{persona}} aliases; feed the user's real name into {{user}} instead of the literal 'User').
<!-- SECTION:NOTES:END -->
