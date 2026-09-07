---
id: TASK-431
title: File picker card filter and start-location defaults
status: Done
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-22 07:05'
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
- [x] #1 Card import filter accepts .webp; .md is removed from the default card filter or clearly ranked/marked so docs do not read as cards
- [x] #2 When a filter hides entries, the listing says how many were hidden
- [x] #3 The import picker remembers a per-context start directory (last successful character import location) instead of a global last location
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Card-import filter + start-location fixes.
AC#1 (LOCAL, personas_screen.py): extracted _character_import_filters() — 'Character Cards' primary tester now accepts .json/.png/.webp (importer supports webp) and NO LONGER matches .md (docs stopped reading as cards); dedicated Markdown + Card Images (PNG/WebP) sub-filters kept; all testers stay callables.
AC#2 (shared picker, additive): SearchableDirectoryNavigation counts entries excluded specifically by the active file_filter (reusing vendored hide()'s filter-check, guarded vs dotfile/dir double-count), posts FilterHiddenCountChanged → Static#filter-hidden-notice 'N hidden by filter' (blank at 0), mirroring the search-count mechanism.
AC#3: the per-context start-dir infra already existed (filepicker.last_dir_{context}); added a regression test proving contexts stay independent and a saved value is reused (RED-checked by hardcoding the key).
LIVE-VERIFIED: valid_card.webp listed, README/CHANGELOG/NOTES.md hidden, '3 hidden by filter' shown.
Follow-ups noted: CCPCharacterHandler.handle_import is dead code (candidate deletion); multi-select double-click quirk (multi-select is a non-goal).
<!-- SECTION:NOTES:END -->
