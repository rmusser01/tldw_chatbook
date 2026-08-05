---
id: TASK-464
title: Character import: accept integer position in embedded character_book
status: To Do
assignee: []
created_date: '2026-07-21 22:00'
labels:
  - roleplay
  - import
  - lore
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during TASK-429 live verification. A V2 character card whose embedded character_book has an INTEGER position on an entry (e.g. position: 0 / 1 — a common SillyTavern world-info convention) fails validate_character_book_entry (Character_Chat_Lib.py: optional_fields_entry["position"] = str + valid_positions string-enum check), which aborts the ENTIRE card import ("Card explicitly declared as V2 but failed V2 structural validation. Import aborted.") before TASK-429's lorebook conversion ever runs. Spec-compliant V2 uses string positions ("before_char"/"after_char"), so strictly-compliant cards import fine, but many real-world SillyTavern exports use int positions and are rejected wholesale. The TASK-429 converter already coerces int->string position via world_book_import._normalize_position, so relaxing the validator (coerce int position to the string enum, or accept the known int->string mapping) would let these real cards import AND get their lorebook converted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A V2 card whose character_book entry uses an integer position (0/1) imports successfully instead of aborting the whole card
- [ ] #2 The integer position is coerced to the correct string enum on the imported/converted world-book entry
- [ ] #3 Strictly-compliant string-position cards continue to import unchanged
<!-- AC:END -->
