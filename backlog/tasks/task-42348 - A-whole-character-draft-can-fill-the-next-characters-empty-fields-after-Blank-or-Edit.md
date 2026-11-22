---
id: TASK-42348
title: "A whole-character draft can fill the next character's empty fields after Blank or Edit"
status: To Do
assignee: []
created_date: 2026-11-22 16:37
updated_date: 2026-11-22 16:37
labels:
- follow-up
- roleplay
- bug
priority: high
references:
- tldw_chatbook/UI/Screens/personas_screen.py
- tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py
- Tests/UI/test_personas_character_generation_ui.py
- Tests/UI/test_personas_generation_wiring.py
dependencies:
- TASK-33910.50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of the TASK-33910.50 port (PR #3431, round 20, fu/r20-roleplay). Read at 59cc99e61b (dev 8a55555798 plus the round-20 stack); line numbers are that tree's. Found by reading; not reproduced.

**Why it matters.** An author who asks for a whole-character draft and then starts another character (Blank, From a portrait, From a concept) or opens one with Edit before the draft returns gets the draft written into the other character's empty fields. If they save, that character's card is wrong. This is silent data corruption of a saved character.

**Evidence.**
- `_run_whole_character_generation` (`personas_screen.py:11088-11113`) captures the editor, awaits `generate_whole_character`, then calls `editor.apply_generated_character(fields)` with no check that the editor still holds the character the draft was asked for. The editor is one widget reused across characters: `new_character` and `load_character` reload it in place (`personas_character_editor_widget.py:749`, `:870`).
- The worker group `"character-generation"` (`personas_screen.py:11082-11086`, `:11126-11130`, `:11142-11146`) is never cancelled when a new character session starts; the screen cancels only its list-window and also-in groups (`:2019-2020`).
- `_run_field_generation` (`:11029`) has the same shape, but it shows a preview the author must accept rather than writing the field, so a late result there is a stray preview, not a silent write.

TASK-33910.50 made a new or loaded character close the "From a concept" row; it did not touch generation still in flight.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A whole-character draft that returns after the author started or opened another character changes none of that character's fields, and the author is not told that fields were filled
- [ ] #2 A single-field generation that returns after the character changed shows no preview on the other character
- [ ] #3 A draft that returns while the same character is still open fills its empty fields as before
- [ ] #4 Mounted tests with a held fake generation controller cover Blank and Edit during a whole-character draft and a character change during a field generation, each shown red before the fix
- [ ] #5 The new tests run on a PR lane
<!-- AC:END -->
