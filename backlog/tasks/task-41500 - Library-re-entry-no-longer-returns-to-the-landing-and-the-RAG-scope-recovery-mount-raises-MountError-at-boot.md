---
id: TASK-41500
title: "Library re-entry no longer returns to the landing and the RAG scope recovery mount raises MountError at boot"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- library
- ui
priority: high
references:
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_screen_navigation.py
dependencies:
- TASK-41127
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41127 (libtests, round 15) and left red as product defects. Library screen reuse (7992124e8d) skips `restore_state` (`library_screen.py:10650`), so re-entering Library no longer returns to the landing. Separately `_mirror_library_rag_scope_recovery` (`library_screen.py:37292`) mounts into `#library-rag-source-scope` before it is mounted and the worker raises `MountError`.

**Evidence.** `test_generic_reentry_returns_to_library_landing` and `test_library_screen_round_trip_returns_to_landing_with_rag_draft` fail 3 of 3 (twice on `'browse-search' == ''`, once on the RAG `MountError`). `test_boot_with_search_default_tab_lands_on_library_rag_canvas` fails on the `MountError`. Turning reuse off makes the first test pass (41127 notes). Re-confirmed in the libtests fix round.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Leaving Library and returning through any route lands on the Library landing, not on the previous search or rail state, unless a route explicitly asks to restore
- [ ] #2 Booting with the Search default tab reaches the RAG canvas with no `MountError` from the scope recovery worker
- [ ] #3 The three named nodes pass, with the red-before evidence recorded
- [ ] #4 A reuse path that skips `restore_state` is either removed or has a test pinning what it intentionally keeps
<!-- AC:END -->
