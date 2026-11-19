---
id: TASK-38228
title: 'Character window-read wrappers swallow database errors'
status: Done
assignee:
  - '@claude'
created_date: '2026-10-30 18:43'
updated_date: '2026-11-05 18:30'
labels:
  - character-chat
  - bug
dependencies: []
references:
  - tldw_chatbook/Character_Chat/Character_Chat_Lib.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
get_character_page_for_ui and count_character_page log a database error and return [] and 0, so their callers cannot tell a failed read from an empty one. It exists on dev today; Roleplay frame B2b's failure row with Retry therefore cannot show for a database fault inside them (the list reads as "no match"), and B2b does not fix the wrappers. Found during Roleplay frame B2b (TASK-33910.4, plan Docs/superpowers/plans/2026-10-26-roleplay-b2b-toolbar-new-import.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A database error in a character window read reaches its caller as an error, not as an empty result
- [x] #2 Roleplay shows its failure row with Retry for that error, pinned by a test that injects it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in burn-down PR 3 of the B2b follow-ups ("list failures read as failures"), with TASK-33910.46 and TASK-33910.37.

- **On dev (29592ecc9b):** `get_character_page_for_ui`, `count_character_page` and `list_character_tags` (`Character_Chat/Character_Chat_Lib.py`) caught every exception, logged it, and answered `[]` / `0` / `[]`. A database fault in Roleplay's character list therefore read as an empty library: "No characters yet - use New or Import to add one.", "Characters 0", Retry hidden. This PR fixes it.
- **Fix:** the three wrappers no longer catch, so a `CharactersRAGDBError` reaches the caller. Their docstrings name the raise. The file stays at 4,775 lines, so its ratchet row is unchanged. The three `error` log statements went with the catches (both callers log the exception themselves); the diagnostic inventory was re-read and re-written for that file.
- **Callers:** both production callers are in `personas_screen.py`, and both already catch and log, so no caller code changed.
  - The window read (`_list_window`) falls into its existing `except Exception`: the B2b failure row with Retry, the "Characters (?)" title, and the fixed toast "Could not load characters.".
  - The tag picker (`_tag_filter_worker`) toasts the fixed "Could not read character tags." where dev opened an empty picker.
  - Both toasts are fixed text, and both logs give the exception's category only.
    - Dev already interpolated the exception text into both toasts and logged the traceback. On dev no database fault ever reached them; other errors did.
    - Once the wrappers raised, the text would have shown SQL ("Query execution failed: no such table: character_cards") or, for a failed connection, the database file's path.
    - PR #3427 review fix round 1 fixes it. `test_a_database_fault_toasts_fixed_text_never_the_database_error` pins it for both reads, and mutations restore the interpolation, name the path, and change the wording.
    - The two re-worded log statements were read with `--statements` and re-pinned in the diagnostic inventory.
  - The Also-in count already called `db.count_character_cards` directly.
- **Tests:**
  - AC#1: `Tests/DB/test_character_cards_paging.py::test_lib_wrappers_raise_a_database_fault_instead_of_an_empty_result`, 5 cases. It uses a real `:memory:` ChaChaNotes, and a SQLite authorizer denies reads of `character_cards`.
  - AC#2: `Tests/UI/test_roleplay_item_list_app.py::test_a_database_fault_shows_the_recovery_row_and_the_centre_says_so_until_retry`. It uses a real file-backed ChaChaNotes whose `character_cards` table is renamed away and then back for Retry. `:memory:` cannot serve here: the screen reads off-thread, one connection per thread (`roleplay_frame_harness.character_db_in_use`).
  - Both are RED on dev for the right reason (DID NOT RAISE; list state `empty`, not `recovery`). Every new assertion is shown red by a named mutation (see the PR report).
- **Not fixed (exists on dev):** `get_character_list_for_ui` in the same file still answers `[]` on an error. It is the dropdown loader, not a window read, and the triage does not name it.
<!-- SECTION:NOTES:END -->
