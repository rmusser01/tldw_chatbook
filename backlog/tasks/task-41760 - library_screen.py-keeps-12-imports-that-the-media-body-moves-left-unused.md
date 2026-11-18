---
id: TASK-41760
title: "library_screen.py keeps 12 imports that the media body moves left unused"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- architecture
priority: low
references:
- tldw_chatbook/UI/Screens/library_screen.py
- pyproject.toml
- Tests/Architecture/test_library_support_layer_surface.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the second re-review of TASK-36952.3.8 (finding R2-1, 2026-11-18). The census counts a module-level import of `library_screen.py` as unused when the module never loads the name and no string in the module names it. It finds 42 such imports at dev 45b1949ca3, the series' base, and 54 at dev 908e9840aa (2026-11-18). The 12 new ones are what the moves left behind:
- `suppress` (`:24`, from contextlib);
- `LibraryMediaReaderSessionState` and `set_more_open` (`:190`);
- `LibraryMediaContentBody` (`:450`);
- `_ANALYZE_AUTO_SKIP_REASON`, `_ANALYZE_ITEM_FAILED_REASON`, `_ANALYZE_SELECTED_WORKER_GROUP`, `_MEDIA_PROGRESS_RESTORE_LAYOUT_SIGNALS`, `_TRASH_PERMANENT_DELETE_FAILURE_COPY` and `_TRASH_RESTORE_FAILURE_COPY` (`:684`);
- `_LibraryMediaSettlementOutcome` and `_LibraryMediaSuccessfulFocusOwnership` (`:770`).

Ruff cannot report them: `pyproject.toml:730` turns F401 off for this file, because the file re-exports its support layer. `Tests/Architecture/test_library_support_layer_surface.py` pins two of the 12 as re-exports: `_LibraryMediaSettlementOutcome` and `_LibraryMediaSuccessfulFocusOwnership`. No test or module reads the other ten through `library_screen`.

The series' own convention was to drop the imports a move orphans: 3.1 removed 6, 3.2 removed 1, 3.7 removed `ascii_glyph_mode` and 3.8 removed `Coroutine`. These 12 slipped through.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the 12 is removed from library_screen.py, or kept with a recorded reason naming the reader that needs it
- [ ] #2 LibraryScreen's _BUDGETS row is re-pinned to its measurement
- [ ] #3 The support-layer surface, undefined-globals and Library preimport closure and payload tests pass
<!-- AC:END -->
