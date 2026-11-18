---
id: TASK-41758
title: "Split the 11,085-line media controller by cluster"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- architecture
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_media_controller.py
- Tests/Architecture/test_library_modules_size_ratchet.py
- backlog/docs/library-decomposition-recipe.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The media body-move series (TASK-36952.3.1 to 3.8, merged 2026-11-18) moved 142 bodies from LibraryScreen into `library_media_controller.py`. Its rulings had each slice raise the controller's size row rather than stall ("line counts are re-pins, never stalls"), and asked for a follow-up to split the controller by cluster once the series landed.

Measured with the size ratchet's own line count, the module grew from 5,313 lines at dev 45b1949ca3 (the series' base) to 11,085 at dev 908e9840aa (2026-11-18). The `LibraryMediaController` class is 10,214 lines with 397 methods, and the module docstring alone is 630 lines. The next-largest Library controller, `library_notes_controller.py`, is 7,462 lines.

The slices' clusters give a ready-made split:
- find and the Console handoff;
- the mutation interlock, delete and edit;
- select mode, browse and take-note;
- Trash;
- analysis;
- detail, reading progress and speaker rename;
- return settlement, reader layout and review actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No media controller module is longer than 5,313 lines, the controller's size before TASK-36952.3
- [ ] #2 The moves are pure: moved bodies are byte-identical apart from tokens the notes list, and the move commits are in .git-blame-ignore-revs
- [ ] #3 The screen, the canvases, the other controllers and the tests reach media behaviour by unchanged names, or each reach is retargeted in the same change with its assertions unchanged
- [ ] #4 Each new module has a row in Tests/Architecture/test_library_modules_size_ratchet.py at its measurement, and LibraryScreen's _BUDGETS row and _LIBRARY_INIT_LAMBDAS do not rise
<!-- AC:END -->
