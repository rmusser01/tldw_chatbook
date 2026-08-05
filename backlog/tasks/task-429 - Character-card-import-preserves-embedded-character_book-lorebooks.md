---
id: TASK-429
title: Character card import preserves embedded character_book lorebooks
status: Done
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-22 04:24'
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
- [x] #1 Importing a v2 card with a character_book results in that lorebook existing in the app and being attached to the imported character
- [x] #2 Import surfaces what happened to the lorebook (e.g. toast naming the imported book); if any part cannot be imported, the user is told instead of silent dropping
- [x] #3 Round-trip: exporting the character and re-importing preserves the book
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Character-card import now preserves an embedded V2 character_book: parse_v2_card converts it into the app's managed extensions['character_world_books'] snapshot (so it is visible/attached in the character's World Books panel and injects at send-time), and drops the legacy extensions['character_book'] key to avoid double-injection (the send path unions both with no cross-dedup). No ChaChaNotes migration (extensions JSON blob, v22); no export changes (extensions round-trips verbatim → AC#3).

3 TDD tasks: (1) lenient converter character_book_to_world_book_block in world_book_import.py reusing _normalize_entry (salvage-and-count, never raises); (2) parse_v2_card converts from the top-level V2 field OR the nested legacy key, merges deduped-by-name, and pops character_book ONLY after a block with >=1 salvaged entry (never pops unconverted lore — closes a legacy round-trip regression); (3) Personas import toast names the book + entry count via a fetch_character_by_id re-read, and an honest name-conflict toast.

Design/adversarial-review catches: legacy characters store the book NESTED (exported verbatim, no top-level field) so re-importing an exported legacy card must convert-from-nested-then-pop or it would delete working lore (AC#3 regression); handling nested also upgrades legacy cards on re-import. Embedded-snapshot only (no standalone world_books row — UNIQUE-name conflict-prone). Re-import/update-existing scoped out (pre-existing whole-card no-update; honest toast).

Whole-branch opus review: ready to merge, no Critical/Important. Tests: test_world_book_import (32), test_character_file_operations (13), full Tests/Character_Chat 374 + 1 pre-existing unrelated failure, Tests/UI/test_personas_workbench 170; 57 send-path/resolve tests confirm single injection. LIVE-VERIFIED via the real import code path: World Books panel shows 'Second Chance Lore — 2 entries' attached, DB has no legacy character_book key (no double-inject), real export→reimport preserves the book without duplication.

Pre-existing gap found (out of scope, follow-up filed): a V2 card with an INT position in character_book fails validate_character_book_entry (requires a string enum) and aborts the whole import before conversion — the converter handles int for robustness but the validation gate blocks first.

Files: tldw_chatbook/Character_Chat/world_book_import.py, tldw_chatbook/Character_Chat/Character_Chat_Lib.py, tldw_chatbook/UI/Screens/personas_screen.py + their tests.
<!-- SECTION:NOTES:END -->
