---
id: TASK-431
title: File picker card filter and start-location defaults
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - widgets
  - ux
  - file-picker
  - roleplay
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live: the "Character Cards" filter includes .md so a docs-heavy folder lists every README/CHANGELOG as an importable card, while .webp cards are hidden even though the importer supports webp (personas_screen.py:3734 vs Character_Chat_Lib.py:2067). A filtered folder can render as just ".." with no hint that entries are hidden. The picker also reopens at the last location used by ANY screen (observed: /Applications) rather than a stable, context-relevant default, and Recent tracks only files so it stays empty until a first success.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Card import filter accepts .webp; .md is removed from the default card filter or clearly ranked/marked so docs do not read as cards
- [ ] #2 When a filter hides entries, the listing says how many were hidden
- [ ] #3 The import picker remembers a per-context start directory (last successful character import location) instead of a global last location
<!-- AC:END -->
