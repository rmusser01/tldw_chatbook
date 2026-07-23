---
id: TASK-429
title: Character card import preserves embedded character_book lorebooks
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - roleplay
  - import
  - lore
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live: importing a Character Card v2 containing a character_book ("Second Chance Lore") silently dropped the lorebook - the character shows "No world books attached" and Lore mode shows "No lore books yet". Every other card field survived (tags, alternate greetings, embedded avatar, prose fields). For SillyTavern-style cards the lorebook is behavior-critical; the character then plays without its lore with zero warning. The P2f embedded-snapshot seam (character_cards.extensions['character_world_books']) exists but the import path never populates it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Importing a v2 card with a character_book results in that lorebook existing in the app and being attached to the imported character
- [ ] #2 Import surfaces what happened to the lorebook (e.g. toast naming the imported book); if any part cannot be imported, the user is told instead of silent dropping
- [ ] #3 Round-trip: exporting the character and re-importing preserves the book
<!-- AC:END -->
