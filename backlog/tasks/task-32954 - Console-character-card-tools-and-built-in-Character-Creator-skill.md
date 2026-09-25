---
id: TASK-32954
title: Console character card tools and built-in Character Creator skill
status: To Do
created_date: 2026-09-25 18:23
labels:
- characters
- agents
- skills
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users want to ask the Console assistant to create a new character card or update an existing one, with a Character Creator skill guiding the conversation. The skill ships with the app and the tools are on by default; every save asks for approval. Design: Docs/superpowers/specs/2026-09-25-character-card-tools-design.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user can ask the Console assistant to create a character; after an interview and a draft they approve, the character is saved locally and can be chatted with
- [ ] #2 A user can ask to change specific fields of an existing character; only those fields change, and a stale or partially-read card is never overwritten
- [ ] #3 A save can attach an avatar generated through the configured Image Gen backend or read from a local file; an avatar failure never loses the text
- [ ] #4 Every save shows an approval card that summarises the change without exposing full field text
- [ ] #5 The Character Creator skill is available on a fresh install without skill-trust setup, can be disabled, and can be customised into the user's own copy
- [ ] #6 In a server-mode Console session the tools refuse clearly instead of writing locally
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
